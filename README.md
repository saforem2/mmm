# `mmm`: Multi-Model Models

## 🐣 Getting Started

### 📝 Example

```bash
git clone https://github.com/saforem2/mmm
cd mmm

export PBS_O_WORKDIR=$(pwd)
source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
ezpz_setup_env

python3 -m pip install -e . --require-virtualenv

# ---- [smoke-test] ------------------------------------
#test ability to launch simple distributed training job:
launch python3 -m ezpz.test_dist

# ---- [FSDP Example] -----------------------------------
launch python3 \
  src/mmm/examples/fsdp/example.py \
  --lr 1e-4 \
  --epochs 20 \
  --batch-size 1024 \
  --dtype bf16 \
  --gamma 0.99
```

<details closed><summary>Output:</summary>

```bash
# e.g. on Sunspot:
$ launch python3 -Wignore src/mmm/examples/fsdp/example.py --lr 1e-4 --batch-size 128 --epochs 10
Disabling local launch: multi-node application
Connected to tcp://x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com:7919
Found executable /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application 515ccad8-5eee-498e-8782-74612e854f7c
[2024-12-04 08:29:36,983][INFO][ezpz.dist]: [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
[2024-12-04 08:29:36,987][INFO][ezpz.dist]: [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-12-04 08:29:37,228][INFO][ezpz.dist]: [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
[2024-12-04 08:29:37,230][INFO][ezpz.dist]: [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
[2024-12-04 08:29:37,231][INFO][ezpz.dist]: [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-12-04 08:29:37,233][INFO][ezpz.dist]: [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-12-04 08:29:37,233][INFO][ezpz.dist]: [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-12-04 08:29:37,239][INFO][ezpz.dist]: [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
[2024-12-04 08:29:37,240][INFO][ezpz.dist]: [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
[2024-12-04 08:29:37,240][INFO][ezpz.dist]: [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
[2024-12-04 08:29:37,242][INFO][ezpz.dist]: [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-12-04 08:29:37,243][INFO][ezpz.dist]: [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-12-04 08:29:37,247][INFO][ezpz.dist]: [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
[2024-12-04 08:29:37,280][INFO][ezpz.dist]: [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-12-04 08:29:37,286][INFO][ezpz.dist]: [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
[2024-12-04 08:29:37,289][INFO][ezpz.dist]: [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-12-04 08:29:37,331][INFO][ezpz.dist]: [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-12-04 08:29:37,332][INFO][ezpz.dist]: [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
[2024-12-04 08:29:37,332][INFO][ezpz.dist]: [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-12-04 08:29:37,360][INFO][ezpz.dist]: [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
[2024-12-04 08:29:37,361][INFO][ezpz.dist]: [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-12-04 08:29:37,362][INFO][ezpz.dist]: [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
[2024-12-04 08:29:37,367][INFO][ezpz.dist]: [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
[2024-12-04 08:29:37,380][INFO][ezpz.dist]:

[dist_info]:
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com', 'x1921c1s2b0n0.hostmgmt2000.cm.americas.sgi.com']
  • HOSTFILE=/var/spool/pbs/aux/10284362.amn-0001
  • HOSTNAME=x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com
  • LOCAL_RANK=0
  • MACHINE=SunSpot
  • NUM_NODES=2
  • NGPUS=24
  • NGPUS_AVAILABLE=24
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=24
  • WORLD_SIZE_IN_USE=24
  • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/10284362.amn-0001 --cpu-bind depth -d 16


[2024-12-04 08:29:37,380][INFO][ezpz.dist]: Using oneccl_bindings from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-12-04 08:29:37,380][INFO][ezpz.dist]: Using ipex from: /opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-12-04 08:29:37,380][INFO][ezpz.dist]: [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2024-12-04 08:29:37,385][INFO][ezpz.dist]: [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-12-04 08:29:37,385][WARNING][ezpz.dist]: Using [24 / 24] available "xpu" devices !!
2024:12:04-08:29:37:(158466) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
[2024-12-04 08:29:38,578][INFO][__main__]: model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2024-12-04 08:29:51,996][INFO][__main__]: epoch=1 dt=12.933120 train_loss=1.801742 test_loss=1.092526 test_acc=80.725418
[2024-12-04 08:29:52,931][INFO][__main__]: epoch=2 dt=0.579176 train_loss=0.975217 test_loss=0.607414 test_acc=86.820541
[2024-12-04 08:29:53,894][INFO][__main__]: epoch=3 dt=0.674108 train_loss=0.689858 test_loss=0.469025 test_acc=89.158676
[2024-12-04 08:29:54,953][INFO][__main__]: epoch=4 dt=0.709618 train_loss=0.588500 test_loss=0.412870 test_acc=90.257797
[2024-12-04 08:29:56,152][INFO][__main__]: epoch=5 dt=0.884786 train_loss=0.542037 test_loss=0.382394 test_acc=90.927261
[2024-12-04 08:29:56,937][INFO][__main__]: epoch=6 dt=0.502172 train_loss=0.513188 test_loss=0.364508 test_acc=91.247002
[2024-12-04 08:29:57,840][INFO][__main__]: epoch=7 dt=0.541722 train_loss=0.494552 test_loss=0.353867 test_acc=91.456833
[2024-12-04 08:29:58,702][INFO][__main__]: epoch=8 dt=0.513719 train_loss=0.477829 test_loss=0.344874 test_acc=91.566750
[2024-12-04 08:29:59,632][INFO][__main__]: epoch=9 dt=0.511596 train_loss=0.473746 test_loss=0.340128 test_acc=91.676659
[2024-12-04 08:30:00,589][INFO][__main__]: epoch=10 dt=0.612377 train_loss=0.467006 test_loss=0.336481 test_acc=91.696640
[2024-12-04 08:30:00,589][INFO][__main__]: 11 epochs took 22.0s
[2024-12-04 08:30:01,133][INFO][ezpz.history]: Saving epoch plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,370][INFO][ezpz.history]: Saving dt plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,582][INFO][ezpz.history]: Saving train_loss plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,786][INFO][ezpz.history]: Saving test_loss plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:01,982][INFO][ezpz.history]: Saving test_acc plot to: /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots
[2024-12-04 08:30:02,214][INFO][ezpz.plot]: Appending plot to: plots/tplots/dt.txt
[2024-12-04 08:30:02,219][INFO][ezpz.plot]: Appending plot to: plots/tplots/train_loss.txt
[2024-12-04 08:30:02,225][INFO][ezpz.plot]: Appending plot to: plots/tplots/test_loss.txt

                            dt [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
12.9┤▌                                                                    │
    │▚                                                                    │
    │▝▖                                                                   │
10.9┤ ▚                                                                   │
    │ ▐                                                                   │
    │  ▌                                                                  │
 8.8┤  ▐                                                                  │
    │   ▌                                                                 │
 6.7┤   ▚                                                                 │
    │   ▝▖                                                                │
    │    ▚                                                                │
 4.6┤    ▐                                                                │
    │     ▌                                                               │
    │     ▐                                                               │
 2.6┤      ▌                                                              │
    │      ▚                                                              │
    │      ▝▖                                                             │
 0.5┤       ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
dt                                   epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/dt.txt

                        train_loss [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
1.80┤▌                                                                    │
    │▝▖                                                                   │
    │ ▚                                                                   │
1.58┤  ▌                                                                  │
    │  ▝▖                                                                 │
    │   ▚                                                                 │
1.36┤    ▌                                                                │
    │    ▝▖                                                               │
1.13┤     ▚                                                               │
    │      ▌                                                              │
    │      ▝▖                                                             │
0.91┤       ▝▄                                                            │
    │         ▀▄                                                          │
    │           ▀▄                                                        │
0.69┤             ▀▄▖                                                     │
    │               ▝▀▀▄▄▖                                                │
    │                    ▝▀▀▚▄▄▄▄▄▄▄                                      │
0.47┤                               ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
train_loss                           epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/train_loss.txt

                         test_loss [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
1.09┤▌                                                                    │
    │▝▖                                                                   │
    │ ▚                                                                   │
0.97┤  ▌                                                                  │
    │  ▝▖                                                                 │
    │   ▚                                                                 │
0.84┤    ▌                                                                │
    │    ▝▖                                                               │
0.71┤     ▚                                                               │
    │      ▌                                                              │
    │      ▝▖                                                             │
0.59┤       ▝▄                                                            │
    │         ▀▄                                                          │
    │           ▀▄                                                        │
0.46┤             ▀▄▖                                                     │
    │               ▝▀▀▀▚▄▄▄▖                                             │
    │                       ▝▀▀▀▄▄▄▄                                      │
0.34┤                               ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
test_loss                            epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/test_loss.txt
[2024-12-04 08:30:02,237][INFO][__main__]: dataset=<xarray.Dataset> Size: 360B
Dimensions:     (draw: 10)
Coordinates:
  * draw        (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables:
    epoch       (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt          (draw) float64 80B 12.93 0.5792 0.6741 ... 0.5137 0.5116 0.6124
    train_loss  (draw) float32 40B 1.802 0.9752 0.6899 ... 0.4778 0.4737 0.467
    test_loss   (draw) float32 40B 1.093 0.6074 0.469 ... 0.3449 0.3401 0.3365
    test_acc    (draw) float32 40B 80.73 86.82 89.16 90.26 ... 91.57 91.68 91.7
                         test_acc [2024-12-04-083002]
    ┌─────────────────────────────────────────────────────────────────────┐
91.7┤                                      ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
    │                            ▄▄▞▀▀▀▀▀▀▀                               │
    │                       ▄▄▞▀▀                                         │
89.9┤                  ▄▄▞▀▀                                              │
    │              ▗▀▀▀                                                   │
    │            ▗▞▘                                                      │
88.0┤          ▗▞▘                                                        │
    │        ▗▞▘                                                          │
86.2┤       ▞▘                                                            │
    │      ▞                                                              │
    │     ▐                                                               │
84.4┤    ▗▘                                                               │
    │   ▗▘                                                                │
    │   ▞                                                                 │
82.6┤  ▞                                                                  │
    │ ▐                                                                   │
    │▗▘                                                                   │
80.7┤▌                                                                    │
    └┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬──────┬───────┬┘
     1       2      3       4      5       6      7       8      9      10
test_acc                             epoch
text saved in /lus/gila/projects/Aurora_deployment/foremans/projects/saforem2/mmm/plots/tplots/test_acc.txt
Application 515ccad8 resources: utime=1232s stime=259s maxrss=2950696KB inblock=220230 oublock=640 minflt=14481515 majflt=544754 nvcsw=637540 nivcsw=251406
took: 0h:00m:35s
```

</details>
