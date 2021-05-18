# sssp

## Installation and Usage

See `./install.sh` for installation and usage.

## Misc. Results

## MGPU results (EC2 V100)

| dataset | n_gpu | time (ms) |
| ------- | ----- | --------- |
rmat20    | 1     |   3.403
rmat20    | 2     |   3.619
rmat20    | 3     |   3.894
rmat20    | 4     |   4.032
rmat22    | 1     |  27.762
rmat22    | 2     |  20.756
rmat22    | 3     |  18.615
rmat22    | 4     |  17.375
rmat24    | 1     | 239.839
rmat24    | 2     | 143.859
rmat24    | 3     | 113.369
rmat24    | 4     |  97.710

## MGPU+NCCL (EC2 V100)

| dataset | n_gpu | time (ms) |
| ------- | ----- | --------- |
rmat24    | 1     | 238.055
rmat24    | 2     | 140.895
rmat24    | 3     |  96.296
rmat24    | 4     |  72.972

## essentials results (EC2 V100)

| dataset | n_gpu | time (ms) |
| ------- | ----- | --------- |
rmat18    | 1     |   6.39
rmat20    | 1     |  20.50
rmat22    | 1     |  52.52
rmat24    | 1     | 280.73
