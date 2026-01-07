
# DOGI

## What is DOGI?

**DOGI** is an oracle-inspired data placement technique that combines simple yet effective heuristics with lightweight machine learning. DOGI predicts invalidation times for data blocks with high accuracy, dynamically adjusts group configurations, and finds a sweet spot between fine-grained data placement and misprediction penalty.

DOGI is implemented in a prototype log-structured system built on zoned storage (ZNS-SSD), which enables us to measure WAF, I/O performance, and CPU overheads of the system.

The paper that introduces DOGI is currently under revision for [USENIX FAST 2026](https://www.usenix.org/conference/fast26).

An artifact archive snapshot with a detailed screencast used in the paper will be available at **(link)** (not available until **Feb 24, 2026**).


## Implementation Overview

DOGI is implemented on top of [SepBIT’s implementation](https://github.com/fallfish/sepbit), which is a real prototype of a log-structured storage system built on **ZenFS**.

For evaluation, we use real ZNS-SSD devices (Western Digital ZN540 2TB ZNS NVMe SSD).
However, you can also evaluate WAF using an **emulated ZNS SSD** with **NVMeVirt**.


## Hardware Prerequisites

The hardware requirements for executing DOGI are as follows.

* **DRAM**: Must be larger than **(device size of trace files) + 10% of the device size** for data structures and the over-provisioning (OP) region used during trace replay. For example, to run a trace file with a **128GB** device size, you need at least **140GB** of DRAM. Also, for smooth experiments (especially with device emulation), we recommend an **additional 20GB** of free DRAM beyond that.

* **CPU**: To support fast compilation, execution of the DOGI algorithm, and the emulated ZNS SSD, we recommend a CPU with **at least 6 cores**.


## Software Prerequisites

- **OS**: Linux (tested on **Ubuntu 24.04.3 LTS**, kernel **6.14.0-37-generic**)
- **RocksDB**: `v6.25.3`
- **ZenFS**: `v0.2.0`  
- **NVMeVirt**: required for ZNS SSD emulation (used in our prototype setup)

To run the DOGI prototype, your environment must support **RocksDB**, **ZenFS**, and **NVMeVirt**.  


The setup process for these components is described below.

## Installation & Compilation

Since our experiments require modifying GRUB and inserting kernel modules via insmod, we highly recommend running them in a virtual machine environment.
There are three major steps to install and run the DOGI prototype:

1. Build the **ZenFS environment**
2. Build **ZNS-SSD emulation** using **NVMeVirt**
3. Build the **DOGI environment**

### 0. Install Dependencies

Install system packages required by RocksDB, ZenFS, and the DOGI prototype:

```bash
sudo apt install git build-essential libzbd-dev zbd-utils libgflags-dev pkg-config libopenblas-dev libsnappy-dev zlib1g-dev libbz2-dev libgoogle-perftools-dev liblz4-dev libzstd-dev nvme-cli python3.12-venv cmake
```

### 1. Build ZenFS Environment

#### 1-1. Clone and prepare RocksDB

```bash
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
git checkout v6.25.3
git branch
```

You should see:

```text
* (HEAD detached at v6.25.3)
  main
```

which confirms that you are on `v6.25.3`.

#### 1-2. Clone and configure ZenFS as a RocksDB plugin

From the `rocksdb` directory:

```bash
git clone https://github.com/westerndigitalcorporation/zenfs plugin/zenfs
cd plugin/zenfs/
git checkout v0.2.0
git branch
cd ../../
```

#### 1-3. Build RocksDB with ZenFS plugin

```bash
export DEBUG_LEVEL=0
export ROCKSDB_PLUGINS=zenfs
export DISABLE_WARNING_AS_ERROR=1
export EXTRA_CXXFLAGS="-Wno-error=unused-parameter -include cstdint"
make -j48 db_bench

sudo -E make install
```

#### 1-4. Build ZenFS utility

```bash
cd plugin/zenfs/util
make
```

At this point, RocksDB and ZenFS should be built and installed, and the `zenfs` utility should be compiled.

### 2. Build ZNS-SSD Emulation using NVMeVirt

We use **NVMeVirt** to emulate a ZNS SSD device. Detailed instructions can be found at:
[https://github.com/snu-csl/nvmevirt](https://github.com/snu-csl/nvmevirt)

For our prototype, we use an **8GB device** for ZNS emulation. To support this, we reserve about **12GB** of memory for NVMeVirt.

#### 2-1. Configure GRUB (memmap and CPU isolation)

Edit `/etc/default/grub` and modify:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nopat"
GRUB_CMDLINE_LINUX="memmap=12G\\\$10G isolcpus=0,1 intremap=off"
```

Then update GRUB and reboot:

```bash
sudo update-grub
sudo reboot
```

#### 2-2. Clone and configure NVMeVirt

```bash
git clone https://github.com/snu-csl/nvmevirt
cd nvmevirt
```

Edit **Kbuild** to enable ZNS:

```text
#CONFIG_NVMEVIRT_NVM := y
#CONFIG_NVMEVIRT_SSD := y
CONFIG_NVMEVIRT_ZNS := y
#CONFIG_NVMEVIRT_KV := y
```

Edit `ssd_config.h` (around line 199) to set the zone size:

```c
//#define ZONE_SIZE GB(2ULL)
#define ZONE_SIZE MB(64)
```

#### 2-3. Build and load the NVMeVirt module

```bash
make
sudo insmod ./nvmev.ko \ memmap_start=10G \ memmap_size=12225M \ cpus=0,1
nvme list
```

From the output of `nvme list`, identify the ZNS device name (e.g., `/dev/nvme0n1`). The device whose **Model** field is `CSL_Virt_MN_01` is the ZNS SSD emulated by NVMeVirt.

### 3. Build DOGI Environment

#### 3-1. Clone DOGI prototype and download test workload

```bash
git clone https://github.com/dgist-datalab/DOGI.git
wget https://zenodo.org/record/10409599/files/test-fio-small
cd DOGI/prototype
```

The details of the trace workloads are described in the MiDAS repository:
[https://github.com/dgist-datalab/MiDAS](https://github.com/dgist-datalab/MiDAS)

#### 3-2. Set up Python virtual environment and ML dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
chmod +x DOGI-Train/model_trainer.py
```

#### 3-3. Configure workload and device paths

Edit `app/global.cc` to configure:

* Logical device size
* Segment size
* OP ratio
* Workload path
* Default ZNS device name

In particular, update:

```cpp
...
char wk_name[128]  = "/home/ae/test-fio-small";
...
const char kZnsDevicePath[] = "/dev/nvme0n1";
const char kZbdDeviceName[] = "nvme0n1";
```

Set `wk_name` to the path where you downloaded `test-fio-small`, and set `kZnsDevicePath` / `kZbdDeviceName` to the correct ZNS device.

#### 3-4. Build DOGI prototype

From `dogi-prototype/prototype`:

```bash
mkdir build
cd build && cmake ..
make
```

If compilation succeeds, the DOGI application binary will be created under `build/app`.


## Execution

To run DOGI, simply execute the following from  
`dogi-prototype/prototype/build`:

```bash
cd build
sudo ./app/app DOGI
````

This runs the full DOGI pipeline — it initializes the environment, replays the workload trace, trains the ML model (if needed), and executes DOGI’s data placement algorithm on top of the log-structured system.
In our test environment, the full execution completes in about **30 minutes**.



## Code Structure

DOGI’s implementation spans the `app` and `src` directories. Below is a compact overview of key components.

* **`app/main.cc`** – Initializes environment, loads and replays workload traces, triggers ML training/inference, orchestrates execution.
* **`app/classifier.cc`** – Classifies blocks into **hot** and **frozen**.
* **`src/placement/dogi.cc`** – Places **non-hot** blocks into appropriate groups based on ML-inferred categories.

**Model Training & Inference**

* **`app/freq_features.cc`** – Collects frequency-related features.
* **`app/model_train.cc`** – Builds datasets and triggers model training.
* **`DOGI-Train/model_trainer.py`** – Trains a 2-layer PyTorch MLP.
* **`app/mlp_inference.cc`** – Performs invalidation-time inference using the trained model.

**Group Configuration & GC**

* **`group_optimizer.cc`** – Determines group configuration and GC relocation policy.
* **`group_config.cc`** – Applies group configuration to the system.
* **`src/selection/dogiselect.cc`** – Selects GC victims to satisfy the target **block invalidation time range (BIR)** under the configured groups.


## Results

During the experiment, you can observe that DOGI trains a prediction model, determines a group configuration using the trained model, and uses the inference results to place blocks into appropriate groups.

The following result is from an example run:
* **Experimental Setup**
```
==============================
             Setup            
==============================
Configuration Info
PlacementName: DOGI
Trace Path: /home/ae/dogi-prototype/prototype/app/workload/test-fio-small
GpThreshold: 0.0910, OpRatio: 0.1000, LogicalSizeGb: 8
==============================
Placement algorithm: DOGI
ZenFS file system created. Free space: 12032 MB
Selection algorithm: DogiSelect
```

- **WAF and throughput statistics**: The prototype periodically prints WAF and throughput statistics. It reports overall WAF and throughput, as well as per-20 GiB user write statistics. For every 20 GiB user write, it shows:
  - WAF and throughput
  - Average valid block ratio during GC for each group
  - Number of segments belonging to each group
  - Number of erased segments
  - Average age of segments selected for GC
```
============== Per 20GiB Info ==============
Per20GiB Throughput: 145.248 MiB/s, Total Throughput: 178.865 MiB
UserWrite: 40.0 GiB
Per20GiB WAF: 2.012, Total WAF: 1.783
Group 0[1]: 0.093 (Erase: 257) (Age: 38783.125)
Group 1[4]: 0.630 (Erase: 87) (Age: 232053.471)
Group 2[4]: 0.725 (Erase: 57) (Age: 543273.544)
Group 3[6]: 0.774 (Erase: 40) (Age: 888594.775)
Group 4[28]: 0.808 (Erase: 64) (Age: 1403450.734)
Group 5[86]: 0.873 (Erase: 139) (Age: 3397599.173)
============================================
```

* **Training information**: Once a sufficient number of samples is collected, DOGI starts model training.
In the current configuration, about 10% of total blocks are sampled, and when approximately 400K samples are collected, training is triggered. During this phase, users can observe per-epoch loss and accuracy:

```
[Trainer] Reached maximum true LBA number. Closing true LBA file
[Trainer] Training dataset collection completed.
[ModelTrainer] Running: ../venv/bin/python3 '../DOGI-Train/model_trainer.py' '../DOGI-Train/0_training.csv'
Model training start
Epoch [1/15] Train Loss: 2.1466, Train Acc: 17.78% | Val Loss: 2.0736, Val Acc: 20.88%
Epoch [2/15] Train Loss: 2.0801, Train Acc: 20.65% | Val Loss: 2.0301, Val Acc: 22.23%
Epoch [3/15] Train Loss: 2.0520, Train Acc: 21.84% | Val Loss: 2.0114, Val Acc: 22.44%
Epoch [4/15] Train Loss: 2.0348, Train Acc: 22.62% | Val Loss: 1.9964, Val Acc: 22.69%
...
Model training completed, and export completed
[MLP] Reloaded trained model from ../DOGI-Train/TrainedModel/Model0
```

* **Group configuration and GC relocation information**: After training completes, the **Group Optimizer** selects a group configuration and GC relocation policy that are expected to provide low WAF.
It reports the predicted WAF, group configuration, and each group’s BIR.
For example, if `m_list=[0,3,4,10]`, then categories `c1–c3` are assigned to `G1`, `c4` to `G2`, and `c5–c10` to `G3`.

It also reports GC relocation behavior, showing how blocks predicted in each category should be reassigned at their first GC event:

```
==== Selected configuration ====
WAF=1.7446 K=3 m_list=[0,2,4,10] BIR=[10,38,577]
GC relocation (category -> first GC group): c1->G2 c2->G2 c3->G3 c4->G3 c5->G3 c6->G3 c7->G3 c8->G3 c9->G3 c10->G3 
[GCONF] applied from group optimizer: K=3 waf=1.7446 m_list=[0,2,4,10] BIR=[10,38,577] (Unit: Segment)
[GCONF] group optimizer success
```

