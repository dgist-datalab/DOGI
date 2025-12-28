// app/model_train.cc

#include "app/model_train.h"

ModelTrainer::ModelTrainer(uint64_t deviceBytes)
    :   Complete_on(0),
        sampled_lba(static_cast<uint32_t>(deviceBytes / 4096)),
        sampled_lba_num(0),
        featuretime_truelba(0),
        number_of_nonhot(0),
        number_of_samples(0),
        number_of_true_lbas(0),
        number_of_true_lba_line(0),
        number_of_buffered_true_lbas(0),
        true_lba_buffer_size(100),
        _max_sample_num(500000),
        _max_true_lba_num(10000000),
        sampling_ratio(10),
        file_index(0),
        _deviceBytes(deviceBytes),
        first_feature_time(0),
        last_feature_time(0)
{
    true_lba_list.reserve(static_cast<uint32_t>(true_lba_buffer_size)); // Reserve space for true LBA list
    max_true_lba_num = _max_true_lba_num;
    _MakingNewFile(0); 
}


ModelTrainer::~ModelTrainer() {
    // Destructor logic if needed
}

void ModelTrainer::SaveTrainingData(std::array<uint64_t, 6>* features, uint64_t featuretime, bool isHot) { 
    if (Complete_on) return;
    if (isHot) {
        _SaveTrueLBAData((*features)[0], featuretime, false);
    } else {
        _SaveTrainingData(features, featuretime);
    }
}

void ModelTrainer::_SaveTrainingData(std::array<uint64_t, 6>* features, uint64_t featuretime) {
    // Sample non-hot blocks based on sampling ratio
    if (number_of_samples < _max_sample_num) {
        if (number_of_nonhot == sampling_ratio) {
            if (first_feature_time == 0) first_feature_time = featuretime;
            // Save feature to file
			_SaveTrueLBAData((*features)[0], featuretime, true);
            feature_file << featuretime << ","
                         << (*features)[0] << ","
                         << (*features)[1] << ","
                         << (*features)[2] << ","
                         << (*features)[3] << ","
                         << (*features)[4] << ","
                         << (*features)[5] << "\n";
            number_of_nonhot=0;
            number_of_samples++;
            if (!sampled_lba.set(static_cast<uint32_t>((*features)[0]))) {
                sampled_lba_num++;
            }
            if (number_of_samples % 50000 == 0) {
                printf("Saved %u samples so far.\n", number_of_samples);
                printf("[feature %llu] lba=%u prevLba=%u interval=%u freq_bits=%u freq_bitcnt=%u seg_freq=%lu\n",
                       (unsigned long long)number_of_samples,
                       static_cast<uint32_t>((*features)[0]),
                       static_cast<uint32_t>((*features)[1]),
                       static_cast<uint32_t>((*features)[2]),
                       static_cast<uint32_t>((*features)[3]),
                       static_cast<uint32_t>((*features)[4]),
                       static_cast<uint64_t>((*features)[5]));
            }
        } else _SaveTrueLBAData((*features)[0], featuretime, false);
        number_of_nonhot++;
    } else if (number_of_samples == _max_sample_num) {
		//close the file
        printf("Reached maximum sample number. Closing feature file.\n");
        last_feature_time = featuretime;
        printf("- First feature time: %llu, Last feature time: %llu\n",
                (unsigned long long)first_feature_time,
                (unsigned long long)last_feature_time);
        printf("- Interval: %llu\n", (unsigned long long)(last_feature_time - first_feature_time));
        feature_file.flush();
        feature_file.close();
        max_true_lba_num += number_of_true_lbas; //adjust max true lba num based on collected true lbas
        number_of_samples++;
		//save only true LBA data
	}
}

void ModelTrainer::_SaveTrueLBAData(uint64_t LBA, uint64_t featuretime, bool force_save) {
    //check force_save -> write to file
    if (number_of_buffered_true_lbas > true_lba_buffer_size) {
        number_of_overflowed_true_lbas++;
        force_save=true;
    }
	if (number_of_samples >= _max_sample_num && number_of_true_lbas > max_true_lba_num) force_save=true;
    if (force_save) {
        //write timestamp and lba list to file
        truelba_file << featuretime_truelba << ", ";
        for (const auto& lba : true_lba_list) {
            truelba_file << lba << ", ";
        }
        truelba_file << "\n";
        //clear true_lba_list
        true_lba_list.clear();
        number_of_buffered_true_lbas = 0;
        featuretime_truelba = featuretime;
        number_of_true_lba_line++;
    }
	if (number_of_samples >= _max_sample_num && number_of_true_lbas >= max_true_lba_num) {
		//close the file
        printf("!!!Reached maximum true LBA number. Closing true LBA file!!!\n");
        printf("- Time after last saved feature: %llu - %llu = %llu\n",
               (unsigned long long)featuretime,
               (unsigned long long)last_feature_time,
               (unsigned long long)(featuretime - last_feature_time));
        last_feature_time = featuretime;
        printf("- First feature time: %llu, Last feature time: %llu\n",
               (unsigned long long)first_feature_time,
               (unsigned long long)last_feature_time);
        printf("- Interval: %llu\n", (unsigned long long)(last_feature_time - first_feature_time));

        // IMPORTANT: feature_file may still be open if we reached true-LBA limit
        // before reaching _max_sample_num. If we don't close it, the next
        // _MakingNewFile() can fail to open *_sampled.csv (especially in release
        // builds where asserts are compiled out).
        if (feature_file.is_open()) {
            feature_file.flush();
            feature_file.close();
        }

        truelba_file.flush();
        truelba_file.close();
		//making true interval in training dataset file
		//re-open file??
        Complete_on = 1; // Indicate completion of collecting dataset
        return;
	}
    //check sampled_lba bitmap
    // Add LBA to true LBA list
    true_lba_list.push_back(static_cast<uint32_t>(LBA));
    number_of_buffered_true_lbas++;
    number_of_true_lbas++;
}

// Collecting sampled dataset done -> make complete training dataset

void ModelTrainer::_MakingNewFile(uint64_t featuretime) {
    // Ensure previous streams are closed before re-opening new paths.
    if (feature_file.is_open()) {
        feature_file.flush();
        feature_file.close();
    }
    if (truelba_file.is_open()) {
        truelba_file.flush();
        truelba_file.close();
    }

    feature_file_path = "../DOGI-Train/"+std::to_string(file_index)+"_sampled.csv";
    truelba_file_path = "../DOGI-Train/"+std::to_string(file_index)+"_truelba";

    //open files
    feature_file.open(feature_file_path, std::ios::out);
    truelba_file.open(truelba_file_path, std::ios::out);

    if (!feature_file.is_open() || !truelba_file.is_open()) {
        printf("[ModelTrainer] Failed to open training output files.\n");
        printf("  feature_file_path=%s (open=%d)\n", feature_file_path.c_str(), feature_file.is_open() ? 1 : 0);
        printf("  truelba_file_path=%s (open=%d)\n", truelba_file_path.c_str(), truelba_file.is_open() ? 1 : 0);
        std::abort();
    }

    feature_file << std::fixed << std::setprecision(0);
    truelba_file << std::fixed << std::setprecision(0);

    // Write headers for feature file
    //feature_file << "Idx,LBA,PrevLBA,Interval,FreqBits,FreqBits2,SegFreq\n";

    featuretime_truelba = featuretime;
}

void ModelTrainer::_CleanUp() {
    // Clean up resources
    
    // Delete dynamically allocated readers to prevent memory leak
    if (rev_feature_reader != nullptr) {
        delete rev_feature_reader;
        rev_feature_reader = nullptr;
    }
    if (rev_truelba_reader != nullptr) {
        delete rev_truelba_reader;
        rev_truelba_reader = nullptr;
    }
    
    // Close any open files
    if (completed_file.is_open()) {
        completed_file.close();
    }

    // Close feature/trueLBA streams if they are still open.
    // This prevents the next _MakingNewFile() from failing to create *_sampled.csv.
    if (feature_file.is_open()) {
        feature_file.flush();
        feature_file.close();
    }
    if (truelba_file.is_open()) {
        truelba_file.flush();
        truelba_file.close();
    }
    
    // Reset bitmap and counters
    sampled_lba = _BitMap(_deviceBytes / 4096);
    sampled_lba_num = 0;
    true_lba_list.clear();
    featuretime_truelba = 0;
    max_true_lba_num = _max_true_lba_num;

    number_of_nonhot = 0;
    number_of_samples = 0;
    number_of_true_lbas = 0;
    number_of_true_lba_line = 0;
    number_of_buffered_true_lbas = 0;
    number_of_overflowed_true_lbas = 0;

    //debugging
    first_feature_time = 0;
    last_feature_time = 0;
    
    // Reset buffer size to initial value (not 0!)
    true_lba_list.reserve(static_cast<uint32_t>(true_lba_buffer_size));
    
    // Clear string (not nullptr!)
    completed_file_path.clear();
    file_index++;

    _MakingNewFile(0);
    Complete_on = 0;
}

void ModelTrainer::MakingMLModel() {
    // Implement logic to complete the training dataset
    // print collected dataset info
    printf("Training dataset collection completed.\n");
    printf("Total sampled non-hot blocks: %u\n", number_of_samples);
    printf("Total true LBAs collected: %u\n", number_of_true_lbas);
    printf("Total unique sampled LBAs: %u\n", sampled_lba_num);
    printf("Total overflowed true LBA buffers: %u\n", number_of_overflowed_true_lbas);

    _CompleteTrainingDataset();

    // Ensure the completed training CSV is fully written before running Python.
    if (completed_file.is_open()) {
        completed_file.flush();
        completed_file.close();
    }

    // Run Python trainer synchronously on the newly created *_training.csv.
    // User request:
    //   script: ../DOGI-Train/model_trainer.py
    //   cmd: python3 model_trainer.py <dataset_path>
    auto shell_quote = [](const std::string& s) -> std::string {
        std::string out;
        out.reserve(s.size() + 2);
        out.push_back('\'');
        for (char c : s) {
            if (c == '\'') {
                out.append("'\\''");
            } else {
                out.push_back(c);
            }
        }
        out.push_back('\'');
        return out;
    };

    const std::string script_path = "../DOGI-Train/model_trainer.py";
    const std::string cmd = "python3 " + shell_quote(script_path) + " " + shell_quote(completed_file_path);
    printf("[ModelTrainer] Running: %s\n", cmd.c_str());
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        printf("[ModelTrainer] Python command failed (rc=%d). Continuing cleanup.\n", rc);
    }

    _CleanUp();
}

void ModelTrainer::_CompleteTrainingDataset() {
    // Implement logic to complete the training dataset
    // construct hash map using true LBA file

    // Open training dataset files for reading and writing completed files
    completed_file_path = "../DOGI-Train/"+std::to_string(file_index)+"_training.csv";
    completed_file.open(completed_file_path, std::ios::out);
    completed_file  << "Idx,LBA,PrevLBA,Interval,FreqBits,FreqBits2,SegFreq,RealInterval\n";

    assert(completed_file.is_open());

    // Read feature and true LBA files in reverse order
    rev_feature_reader = new ReverseLineReader(feature_file_path);
    rev_truelba_reader = new ReverseLineReader(truelba_file_path);

    std::string feature_line;
    std::string truelba_line;

    //Read one feature file first
    //(To check the last timestamp and LBA)
    std::vector<uint64_t> feature_tokens;
    feature_tokens.reserve(8);
    std::string token;
    std::unordered_map<uint32_t, uint64_t> truelba_map;
    truelba_map.reserve(sampled_lba_num);
    complete_max_time=0;
    
retry:
    rev_feature_reader->getline_rev(feature_line);
    std::stringstream ss_feature(feature_line);
    while (std::getline(ss_feature, token, ',')) {
            feature_tokens.push_back(static_cast<uint64_t>(std::stoull(token)));
    }
    if (feature_tokens.size() < 7) {
        printf("! Invalid feature line: %s\n", feature_line.c_str());
        feature_tokens.clear();
        goto retry; 
    }
    //assert(feature_tokens.size() >= 7);
    /*
    printf("Read initial feature line for matching.\n");
    printf("read line: %s\n", feature_line.c_str());
    printf("[Time %llu] lba=%u prevLba=%u interval=%u freq_bits=%u freq_bitcnt=%u seg_freq=%lu\n",
           (unsigned long long)static_cast<uint64_t>(feature_tokens[0]),
           static_cast<uint32_t>(feature_tokens[1]),
           static_cast<uint32_t>(feature_tokens[2]),
           static_cast<uint32_t>(feature_tokens[3]),
           static_cast<uint32_t>(feature_tokens[4]),
           static_cast<uint32_t>(feature_tokens[5]),
           static_cast<uint64_t>(feature_tokens[6]));
    */
    if (FILL_MAXTIME) printf("!!! FILL_MAXTIME is ON !!!\n");
    else printf("!!! FILL_MAXTIME is OFF !!!\n");
    uint64_t feature_time = static_cast<uint64_t>(feature_tokens[0]);
    uint32_t feature_lba = static_cast<uint32_t>(feature_tokens[1]);

    uint32_t truelba_lines_read = 0;
    uint32_t feature_lines_read = 1;
    uint32_t removed_lines = 0;
    uint32_t written_lines = 0;
    // Main loop to read true LBA lines and match with feature lines
    while (rev_truelba_reader->getline_rev(truelba_line)) {
        truelba_lines_read++;
        // Parse true LBA line
        std::vector<uint64_t> truelba_tokens;
        std::stringstream ss_truelba(truelba_line);
        std::string token;
        while (std::getline(ss_truelba, token, ',')) {
            //erase empty tokens
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) truelba_tokens.push_back(static_cast<uint64_t>(std::stoull(token)));
        }
        if (truelba_tokens.size() < 2) {
            printf("! Invalid true LBA line at line %d: %s\n", truelba_lines_read, truelba_line.c_str());
            if (truelba_tokens[0] == feature_time) abort();
            continue;
        }
        assert(truelba_tokens.size() >= 2);

        uint64_t truelba_time = static_cast<uint64_t>(truelba_tokens[0]);
        if (!complete_max_time) complete_max_time = truelba_time;
        if (truelba_time == 0) {
            // Leave this loop!!! (first line)
            // NotUsed
            printf("Encountered truelba_time 0, finish the process.\n");
            printf("- truelba lines read: %d\n", truelba_lines_read);
            printf("- feature lines read: %d\n", feature_lines_read);
            printf("- written lines: %d, removed lines: %d\n", written_lines, removed_lines);
            completed_file.flush();
            completed_file.close();
            break;
        }
        //update unordered map (hashmap)
        for (size_t i = 2; i < truelba_tokens.size(); ++i) {
            if (sampled_lba.get(truelba_tokens[i])) truelba_map[static_cast<uint32_t>(truelba_tokens[i])] = truelba_time+(i-1);
        }
        uint32_t truelba_target = truelba_tokens[1];


        if (truelba_lines_read%50000==0) {
            printf("Processed %d true LBA lines so far.\n", truelba_lines_read);
            printf("- TrueLBA linenum: %d, time: %llu | Feature linenum: %d, time: %llu\n",
                   truelba_lines_read,
                   (unsigned long long)truelba_time,
                   feature_lines_read,
                   (unsigned long long)feature_time);
            printf("- Removed lines so far: %d\n", removed_lines);
            printf("- Written lines so far: %d\n", written_lines);
        }

        //error check
        assert(truelba_time >= feature_time);

        // Parse feature line
        if (truelba_time == feature_time) {
            assert(feature_lba == truelba_target);
            
            if (truelba_map.find(truelba_target) != truelba_map.end()) {
                uint64_t truetime = truelba_map[truelba_target];
                uint64_t true_interval = truetime - feature_time;
                completed_file << feature_time << ","
                               << feature_lba << ","
                               << feature_tokens[2] << ","
                               << feature_tokens[3] << ","
                               << feature_tokens[4] << ","
                               << feature_tokens[5] << ","
                               << feature_tokens[6] << ","
                               << true_interval << "\n";
                written_lines++;
            } else {
                removed_lines++;
                if (FILL_MAXTIME) {
                    //write with max time
                    uint64_t true_interval = complete_max_time - feature_time;
                    completed_file << feature_time << ","
                                   << feature_lba << ","
                                   << feature_tokens[2] << ","
                                   << feature_tokens[3] << ","
                                   << feature_tokens[4] << ","
                                   << feature_tokens[5] << ","
                                   << feature_tokens[6] << ","
                                   << true_interval << "\n";
                    written_lines++;
                }
            }
            truelba_map[truelba_target] = truelba_time;
            feature_tokens.clear();
            //re-read the feature file
            
            feature_lines_read++;
            rev_feature_reader->getline_rev(feature_line); 
            //check if feature_line is valid
            if (feature_line.empty()) {
                //process finished
                printf("No more feature lines to read, ending process.\n");
                printf("- truelba lines read: %d\n", truelba_lines_read);
                printf("- feature lines read: %d\n", feature_lines_read);
                printf("- written lines: %d, removed lines: %d\n", written_lines,
                          removed_lines);
                printf("- Removed lines so far: %d\n", removed_lines);
                printf("- Written lines so far: %d\n", written_lines);
                completed_file.flush();
                completed_file.close();
                return;
            }
            std::stringstream ss_feature(feature_line);
            while (std::getline(ss_feature, token, ',')) {
                feature_tokens.push_back(static_cast<uint64_t>(std::stoull(token)));
            }
            if (feature_tokens.size() < 7) {
                printf("feature line number: %d\n", feature_lines_read);
                printf("feature_line: %s\n", feature_line.c_str());
                printf("feature tokens: %zu\n", feature_tokens.size());
                for (size_t idx = 0; idx < feature_tokens.size(); ++idx) {
                    printf("Token %zu: %lu\n", idx, feature_tokens[idx]);
                }
            }
            assert(feature_tokens.size() >= 7);

            feature_time = static_cast<uint64_t>(feature_tokens[0]);
            feature_lba = static_cast<uint32_t>(feature_tokens[1]);
        } else {
            truelba_map[truelba_target] = truelba_time;
        }
    }
}

