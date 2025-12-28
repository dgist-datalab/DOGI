// app/model_train.h
#ifndef APP_MODEL_TRAIN_H
#define APP_MODEL_TRAIN_H

#include <vector>
#include <string>
#include <cassert>
#include <array>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <unordered_map>
#include <cstdlib>

/*
* Bitmap class to track sampled LBAs for model training
* Each bit represents whether a specific LBA has been sampled.
* @param nbits Total number of bits (LBAs) to track.
*/
class _BitMap {
public:
    explicit _BitMap(uint32_t nbits)
        : nbits_(nbits),
        words_((nbits+63)>>6, 0ULL) {}

    // Set the bit at index i to 1
    inline bool set(std::size_t i) {
        assert(i<nbits_);
        bool ret = (words_[i>>6] & (1ULL<<(i&63))) != 0;
        words_[i>>6] |= (1ULL<<(i&63));
        return ret;
    }    

    // Get the value of the bit at index i
    inline bool get(std::size_t i) const {
        assert(i<nbits_);
        return (words_[i>>6] & (1ULL<<(i&63))) != 0;
    }

    // Get the total number of bits
    std::size_t size() const {
        return nbits_;
    }

private:
    std::size_t nbits_; // Number of bits
    std::vector<uint64_t> words_; // Underlying storage (64-bit words)
};


// Small, memory-friendly reverse line reader implemented inline.
// Reads a file from end to start in fixed-size chunks and returns lines
// one by one (logical order per line preserved, trailing '\n' removed).
class ReverseLineReader {
public:
    explicit ReverseLineReader(const std::string& path, std::size_t buf_size = 64*1024)
        : file_(path, std::ios::binary), file_size_(0), read_pos_(0), buf_size_(buf_size), finished_(false) {
        if (!file_.is_open()) return;
        file_.seekg(0, std::ios::end);
        file_size_ = static_cast<std::uint64_t>(file_.tellg());
        read_pos_ = static_cast<std::int64_t>(file_size_);
    }

    ~ReverseLineReader() { if (file_.is_open()) file_.close(); }

    // non-copyable
    ReverseLineReader(const ReverseLineReader&) = delete;
    ReverseLineReader& operator=(const ReverseLineReader&) = delete;

    // Read previous line into out. Returns false when no more lines.
    bool getline_rev(std::string& out) {
        out.clear();
        if (!file_.is_open() || finished_) return false;

        auto trim_cr = [](std::string& s){ if (!s.empty() && s.back() == '\r') s.pop_back(); };

        // Try to find a newline in the buffered tail first
        while (true) {
            auto pos = tail_.find_last_of('\n');
            if (pos != std::string::npos) {
                out = tail_.substr(pos + 1);
                tail_.resize(pos); // keep data before that newline
                trim_cr(out);
                // skip possible empty trailing line caused by final '\n'
                if (out.empty()) {
                    if (read_pos_ == 0 && tail_.empty()) { finished_ = true; return false; }
                    continue;
                }
                return true;
            }

            // No newline in tail. If we've reached file start, tail_ is the first line.
            if (read_pos_ == 0) {
                if (tail_.empty()) { finished_ = true; return false; }
                out = tail_;
                trim_cr(out);
                tail_.clear();
                finished_ = true;
                return true;
            }

            // Read next chunk from file (move read_pos_ backwards)
            std::size_t to_read = static_cast<std::size_t>(std::min<std::int64_t>(read_pos_, static_cast<std::int64_t>(buf_size_)));
            read_pos_ -= static_cast<std::int64_t>(to_read);
            file_.seekg(read_pos_, std::ios::beg);
            std::string chunk;
            chunk.resize(to_read);
            file_.read(&chunk[0], static_cast<std::streamsize>(to_read));
            std::streamsize got = file_.gcount();
            if (got <= 0) { finished_ = true; return false; }
            if (static_cast<std::size_t>(got) != to_read) chunk.resize(static_cast<std::size_t>(got));

            // Look for newline in chunk without copying tail in front of it.
            auto cpos = chunk.find_last_of('\n');
            if (cpos != std::string::npos) {
                // line = chunk after cpos + existing tail
                out.assign(chunk.data() + cpos + 1, chunk.size() - (cpos + 1));
                out.append(tail_);
                // keep prefix of chunk before the newline as new tail
                tail_.assign(chunk.data(), cpos);
                trim_cr(out);
                if (out.empty()) {
                    if (read_pos_ == 0 && tail_.empty()) { finished_ = true; return false; }
                    continue;
                }
                return true;
            }

            // No newline found in chunk: prepend chunk to tail (rare, only when lines > chunk size)
            // This copies but is uncommon; buf_size_ can be increased if needed.
            tail_.insert(0, chunk);
        }
    }

private:
    std::ifstream file_;
    std::uint64_t file_size_;
    std::int64_t read_pos_;
    std::string tail_;
    std::size_t buf_size_;
    bool finished_;
};



/*
* @brief ModelTrainer class to manage training data for ML model
*
* This class handles the collection and storage of training samples,
* 1. Write training dataset by sampling non-hot blocks
* 2. Maintain a bitmap to track sampled LBAs
* 3. Store true LBA list for training dataset labeling (true LBA = all accessed LBA)
*
* @param deviceBytes Total size of the device in bytes (used to calculate number of LBAs)
*/
class ModelTrainer {
public:
    ModelTrainer(uint64_t deviceBytes);
    ~ModelTrainer();
    char IsCollectingDone() { return Complete_on; }
    /*
    * @brief Save training data for a given feature set and true LBA
    * triggered on each write operation
    * @param features Pointer to an array of features for the current write
    * @param featuretime Timestamp associated with the feature set
    * @param isHot Boolean indicating if the block is classified as hot
    */
    void SaveTrainingData(std::array<uint64_t, 6>* features, uint64_t featuretime, bool isHot);

    void MakingMLModel();
private:
    void _SaveTrainingData(std::array<uint64_t, 6>* features, uint64_t featuretime);
    void _SaveTrueLBAData(uint64_t LBA, uint64_t featuretime,bool force_save = false);
    void _MakingNewFile(uint64_t featuretime);
    void _CleanUp();

    char Complete_on = 0; // Flag to indicate completion of training dataset

    uint64_t _deviceBytes; // Total device size in bytes
    _BitMap sampled_lba; //Bitmap to track sampled LBAs
    uint32_t sampled_lba_num; // Number of sampled LBAs
    std::vector<uint32_t> true_lba_list; // List of true LBAs for training
    uint32_t featuretime_truelba; // Timestamp for first LBA in true_lba_list

    int number_of_nonhot=0; //for sampling, max=sampling ratio
    uint32_t number_of_samples;
    uint32_t number_of_true_lbas;
    uint32_t number_of_buffered_true_lbas; // Number of buffered true LBAs before writing to file
    uint32_t number_of_true_lba_line;
    uint32_t number_of_overflowed_true_lbas;
	int true_lba_buffer_size;
	
    uint32_t _max_sample_num; // Maximum number of samples (for training)
    uint32_t max_true_lba_num; // (can be adjusted during runtime)
    uint32_t _max_true_lba_num; //Maximum number of true LBAs (for training dataset Labeling)
    int sampling_ratio; // Sampling ration for non-hot blocks

    int file_index = 0; // File index for saving multiple files
    std::string feature_file_path; // Path to save feature data 
    std::string truelba_file_path; // Path to save true LBA data
    std::ofstream feature_file; // Output file stream for feature data
    std::ofstream truelba_file; // Output file stream for true LBA data

    //For debugging and verification
    uint64_t first_feature_time = 0;
    uint64_t last_feature_time = 0;


    //For completing training dataset
    void _CompleteTrainingDataset();
    int training_file_index=0;
    std::string completed_file_path;
    std::ofstream completed_file; // Output file stream for completed true LBA
    uint64_t complete_max_time;
    char FILL_MAXTIME = 1; // Flag to write max time for missing entries

    ReverseLineReader* rev_feature_reader = nullptr;
    ReverseLineReader* rev_truelba_reader = nullptr;
};


#endif // APP_MODEL_TRAIN_H
