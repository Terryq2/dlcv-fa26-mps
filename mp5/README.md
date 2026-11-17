## ECE 494 / CS 444: Deep Learning for Computer Vision, Fall 2026, Assignment 5

### Instructions

1. Assignment is due at **11:59 PM on Thursday Dec 3 2026**.

2. See [policies](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026/policies.html)
   on [class website](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026).

3. Submission instructions:
    1. On gradescope assignment called `MP5-code`, upload the following 3 files:
        - Your completed `flow.py`, `flow_unet.py` and `flow_editing.py`. We
          will run tests to evaluate your pytorch code for Q1.1-Q1.3,
          Q2.1-Q2.6, Q3.1 and Q3.3.

       Please also note the following points:
        - Do not compress the files into `.zip` as this will not work.
        - Do not change the provided files names nor the names of the functions
          but rather change the code inside the provided functions and add new
          functions. Also, make sure that the inputs and outputs of the
          provided functions are not changed.
        - The autograder will give you feedback on how well your code did.
        - The autograder is configured with the python libraries: `numpy torch
          torchvision gradescope-utils` only.

    2. On gradescope assignment called `MP5-report`, upload the visualizations
       and your answers for the manually graded questions: Q1.4, Q2.7-Q2.9,
       Q3.2 and Q3.4.

4. Lastly, be careful not to work of a public fork of this repo. Make a private
   clone to work on your assignment. You are responsible for preventing other
   students from copying your work.

5. Change Log:
   - 2026-11-12: Initial version.
   - 2026-11-15: Removed Q4 denoising (old Q4.1, Q4.2, Q4.6) and the CFG
     implementation (old Q4.3, now provided); Q4 renumbered to Q4.1-Q4.5.
     Shortened Q2.5, Q3.8 (now 1 pt), Q3.9 and the Q4 write-ups. Q3: new
     held-out cutoffs, `train_flow.py` never overwrites a run (`--resume` to
     continue), 40-minute runs, and `run_notebook.sbatch` to run Part B and
     Q4 as batch jobs on Delta AI. The Q3.4 test no longer needs Q3.6.
   - 2026-11-16: Q3.7 (now Q2.7) is a controlled experiment with the
     hyperparameters, graded manually in `MP5-report`, instead of an
     autograded held-out loss: you no longer upload `flow_ema.pt`. There is
     no test set anymore: `data/afhq64.pt` has training and validation
     images, and you compare your runs with the validation loss. If you
     prepared the data before, `load_afhq` converts your file automatically.
     Removed the VAE (old Q1.1-Q1.2 and Q2.4, and `vae.py`), and renumbered
     the questions: Q1 is now 2D flow matching (old Q2.1-Q2.3; old Q2.5 is
     now Q1.4), Q2 is flow matching on images (old Q3), and Q3 is editing
     (old Q4). The notebooks are now `Q1_flow_matching_2d.ipynb`,
     `Q2_flow_matching.ipynb` and `Q3_flow_editing.ipynb`, and
     `q3_helpers.py` is now `q2_helpers.py`. The Q3 questions are numbered
     in the order of the notebook: Q3.2 is the SDEdit write-up (was Q3.4),
     and the visual anagram and hybrid image code is Q3.3 and Q3.4 (was
     Q3.2 and Q3.3).
   - 2026-11-17: Removed hybrid images (Q3.4, `hybrid_velocity`). Q3.4 is
     now your own visual anagram (4 pts, manually graded), and replaces the
     old Q3.5 write-up. `MP5-code` is now 18 points.

   **Disclaimer about equations**: Web-based viewers (such as GitHub or GitLab)
   may **silently drop** math symbols (like square roots) in the notebooks.
   View the notebooks locally (VS Code, Jupyter Lab).

### Suggested Development Workflow
Each question has a Jupyter notebook with detailed instructions:
`Q1_flow_matching_2d.ipynb` (Q1), `Q2_flow_matching.ipynb` (Q2) and
`Q3_flow_editing.ipynb` (Q3). You implement the TODOs in the `.py` files, and
use the notebooks to test your code and to make the visualizations for your
report.

1. All the code that you write (Q1.1-Q1.3, Q2.1-Q2.6, Q3.1, Q3.3) can be developed and tested on a
   local machine without a GPU. Each test runs in a few seconds on a laptop,
   see [Testing Your Code](#testing-your-code). Q1 also trains its model
   on a laptop in a few seconds.
2. Q2 trains a flow matching model on 64 x 64 images, and Q3 uses a large
   pretrained text-to-image flow matching model (PixNerd, about 4 GB of GPU
   memory). For these, you will need a GPU. As in MP3 and MP4, we have
   arranged access to GPUs on the Delta AI cluster. See instructions
   [here](./DeltaAI.md). Q2 training, Q2 Part B and the Q3 experiments run as
   batch jobs on Delta AI.
   - You should follow best practices in using the cluster (mentioned in the
     document), e.g. not run anything on the login node. Violating Delta AI
     cluster policies may cause you to be banned from using it, and we will
     not be able to provide any help in that situation.
   - Delta AI is a shared resource that we don't control. We aren't able to
     control how long it will take for your jobs to start running. So, start
     ahead of time and budget for the time that your job is queued but not
     running.
   - You can also use other GPUs, like [Kaggle](https://www.kaggle.com/code),
     [Google Colab Pro](https://colab.research.google.com/signup), or your
     own NVIDIA GPU.

### Setup
#### If you are using the Delta AI cluster
1. **Your MP folder on Delta AI.** You write and test your code on your
   computer, and run the GPU parts (Q2 training, Q2 Part B and Q3) on Delta
   AI, from a copy of the MP folder in your work directory: `/work/nvme/bfdu/<login>/mp5`
   (`<login>` is your Delta AI login). Copy the whole MP folder there once,
   from your computer, in the folder that contains it (use the name of your
   MP folder instead of `mp5`):
   ```bash
   scp -r mp5 <login>@dtai-login.delta.ncsa.illinois.edu:/work/nvme/bfdu/<login>/mp5
   ```
   (If your copy has a `data/` or `runs/` folder, move it out first: you do
   not need them on Delta AI.) Then protect your work directory on Delta AI
   with `chmod 700 /work/nvme/bfdu/<login>` (see `DeltaAI.md`). After that,
   upload only the files that you change on your computer, several at once
   (every `scp` asks for Duo), e.g. before you train in Q2:
   ```bash
   scp flow.py flow_unet.py <login>@dtai-login.delta.ncsa.illinois.edu:/work/nvme/bfdu/<login>/mp5/
   ```
   Run every command on Delta AI (the tests, `sbatch`) in this folder: `cd /work/nvme/bfdu/<login>/mp5`.
2. We have prepared the Q2 dataset for you at
   `/work/nvme/bfdu/cs444/fa26/mp5/data/afhq64.pt`, and downloaded the
   pretrained Q3 model to `/work/nvme/bfdu/cs444/fa26/mp5/hf_cache/`. The
   code uses both automatically when it runs on Delta AI.
3. We also have a python virtual environment with all dependencies installed
   at `/work/nvme/bfdu/cs444/fa26/mp5/venv/`. Activate it before you run
   python yourself (e.g. the tests):
   ```bash
   module load python/3.10.14
   source /work/nvme/bfdu/cs444/fa26/mp5/venv/bin/activate
   ```
   `train_flow.sbatch` activates it by itself, so you do not need to
   activate it to submit training jobs.
4. We also have a slurm script for Q2 training, see `train_flow.sbatch`.

#### If you are not using the Delta AI cluster
1. Install pre-requisites in `requirements.txt`. On Windows with an NVIDIA
   GPU, first install torch with CUDA using the command from
   [pytorch.org](https://pytorch.org/get-started/locally/) (e.g.
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`):
   plain `pip install torch` installs a CPU-only version on Windows.
2. **Q2 dataset**: the Q2 notebook downloads
   [AFHQv2](https://huggingface.co/datasets/huggan/AFHQv2) from Hugging Face
   and prepares it, about 10 minutes. You only need to do this once.
3. **Q3 model**: the first time you load PixNerd, it downloads about 8 GB of
   weights to your Hugging Face cache.

### Testing Your Code
Run all tests at once from the MP folder with
```bash
python -m unittest test_functions -v
```
or a single test, e.g. `python -m unittest test_functions.TestClass.test_flow_interpolate -v`.
The notebooks run the same tests after each question. The tests only need
your code (no GPU, no dataset, no pretrained model), and are the same as the
autograder's.

### Problems
In this programming assignment, we will
1. implement a flow matching model on 2D data,
2. reuse the flow matching code to train a class-conditional flow model on
   images, and
3. edit images with a large pretrained text-to-image flow matching model.

1. **Flow Matching** (`Q1_flow_matching_2d.ipynb`, `flow.py`)

   1.1 [2 pts Autograded] Interpolation along the straight path from noise to
   data, `FlowMatching.interpolate`.
   ```bash
   python -m unittest test_functions.TestClass.test_flow_interpolate -v
   ```
   1.2 [2 pts Autograded] Conditional flow matching loss, `FlowMatching.get_loss`.
   ```bash
   python -m unittest test_functions.TestClass.test_flow_loss -v
   ```
   1.3 [2 pts Autograded] Euler sampling, `FlowMatching.sample`.
   ```bash
   python -m unittest test_functions.TestClass.test_flow_sample -v
   ```
   1.4 [2 pts Manually Graded] Flow matching samples with different numbers of
   function evaluations (NFE).

2. **Flow Matching on Images** (`Q2_flow_matching.ipynb`, `flow_unet.py`).
   You will train a class-conditional `FlowUNet` on
   [AFHQ](https://arxiv.org/abs/1912.01865) (64 x 64 photos of cats, dogs and
   wild animals), reusing your Q1 `interpolate`, `get_loss` and `sample`.

   2.1 [1 pts Autograded] Sinusoidal time embedding, `TimeEmbedding.sinusoidal`.
   ```bash
   python -m unittest test_functions.TestClass.test_time_embedding -v
   ```
   2.2 [2 pts Autograded] FiLM conditioning, `ResBlock.forward` and `FlowUNet.embed`.
   ```bash
   python -m unittest test_functions.TestClass.test_resblock_film -v
   ```
   2.3 [2 pts Autograded] The UNet, `FlowUNet.forward`.
   ```bash
   python -m unittest test_functions.TestClass.test_unet_forward -v
   ```
   2.4 [2 pts Autograded] Loss with label dropout, `ImageFlowMatching.get_loss`.
   ```bash
   python -m unittest test_functions.TestClass.test_image_flow_loss -v
   ```
   2.5 [1 pts Autograded] Exponential moving average of the weights, `update_ema`.
   ```bash
   python -m unittest test_functions.TestClass.test_update_ema -v
   ```
   2.6 [1 pts Autograded] Classifier-free guidance, `ImageFlowMatching.get_velocity`.
   ```bash
   python -m unittest test_functions.TestClass.test_image_cfg_velocity -v
   ```

   **Training.** Once all tests pass, train your models with the provided
   `train_flow.py`. It saves checkpoints, sample grids and its log
   (`log.txt`) to its `--run_dir`. Its default hyperparameters train, but are
   **not tuned**. In Q2.7, you run a **controlled experiment**: pick one
   hyperparameter (see the Training section of the notebook), and train at
   least 3 runs that differ only in its value, including the default value.
   Use at most 10,000 steps per run, and at most 6 runs in total (about 50
   GPU-minutes on Delta AI). Give every run its own run folder. Every run starts a new model: if
   the folder already has a run, the new run goes to `<run_dir>_2`,
   `<run_dir>_3`, ... (the output says which), so you never overwrite a run.
   On Delta AI, queue one job per run with the provided
   [train_flow.sbatch](./train_flow.sbatch), from your MP folder on Delta AI
   on the login node (the jobs train at the same time), after uploading your
   `flow.py` and `flow_unet.py` there. Flags after `train_flow.sbatch` are
   passed on to `train_flow.py`. For example, this experiment on gradient
   clipping (an example of the syntax, not a recommendation) submits 3 jobs:
   ```bash
   for v in 0 0.5 1; do sbatch --export=ALL,RUN_DIR=runs/grad_clip$v train_flow.sbatch --grad_clip $v; done
   ```
   You can monitor the status of your jobs using `squeue -u $USER`, and their
   progress with `tail -f ece494mp5-flow_afhq-<job id>.out`. Each job trains
   for at most 40 minutes. To continue a run that stopped early, add
   `--resume`: it continues the run in `RUN_DIR` with the hyperparameters it
   was started with.
   ```bash
   sbatch --export=ALL,RUN_DIR=runs/grad_clip0.5 train_flow.sbatch --resume
   ```
   Elsewhere, run `python train_flow.py --run_dir runs/grad_clip0.5 --grad_clip 0.5` (or
   the training cell in the notebook), with the same flags. On Delta AI, a
   10,000-step run takes about 8 minutes (on an RTX 3060, about 13 minutes).

   **Your goal.** With the default hyperparameters, a 10,000-step run
   reaches a validation loss of about 0.166. Aim for a best run at about
   **0.156 or lower** within 10,000 steps: its samples in Q2.8 and Q2.9 are
   clearly cats, dogs and wild animals, and sharper than with the defaults.
   This target is not graded and is only a guide (two runs with the same
   hyperparameters can differ by about 0.001). If your first experiment does
   not reach it, you may run a second experiment on another hyperparameter,
   with the first one set to its best value (your best run so far is then
   one of its runs, so it needs only 2 new runs), and report both. If a run
   with the default hyperparameters ends far above 0.166 (e.g. above 0.17),
   something is probably wrong, e.g. a mistake in your Q2 code that the
   tests did not catch.

   **Part B.** Once your runs have finished, list their folders in
   `EXPERIMENT_RUNS` in the first cell of Part B of the notebook (if a run
   went to e.g. `runs/grad_clip0.5_2`, use that). Q2.8 and Q2.9 use the run
   with the lowest validation loss, unless you set `RUN_DIR` to another run.
   If you ran two experiments, see Q2.7 in the notebook. On Delta
   AI, edit the notebook on your computer and upload it to your MP folder on
   Delta AI (e.g. `scp Q2_flow_matching.ipynb
   <login>@dtai-login.delta.ncsa.illinois.edu:/work/nvme/bfdu/<login>/mp5/`), and run it
   as a batch job from the login node (it skips training, and takes about a
   minute):
   ```bash
   sbatch run_notebook.sbatch Q2_flow_matching.ipynb
   ```
   Then download the figures in `outputs/`, from your computer:
   ```bash
   scp -O "<login>@dtai-login.delta.ncsa.illinois.edu:/work/nvme/bfdu/<login>/mp5/outputs/q2_*.png" .
   ```

   2.7 [4 pts Manually Graded] **Controlled experiment.** Part B of the
   notebook plots the training loss and the validation loss of your runs,
   and prints a table with their validation losses: the flow matching loss
   of the EMA weights on 192 validation images (64 per class) that are never
   used for training, with fixed noise and fixed times. Use the validation
   loss to choose your best run, explain how the hyperparameter changes the
   losses, and report the hyperparameters of your best run with the loss
   plots (for each experiment, if you ran two).

   2.8 [1 pts Manually Graded] Samples during training.

   2.9 [2 pts Manually Graded] Samples with different guidance scales.

3. **Editing Images with a Pretrained Flow Matching Model**
   (`Q3_flow_editing.ipynb`, `flow_editing.py`). You will use
   [PixNerd](https://arxiv.org/abs/2507.23268), a 1.2B parameter text-to-image
   flow matching model that generates 512 x 512 images directly in pixel space,
   without any training, following Part A of [Berkeley's CS180 project
   3](https://cal-cs180.github.io/fa26/hw/proj3-flow/parta.html).

   3.1 [1 pts Autograded] Image-to-image editing (SDEdit), `sdedit`.
   ```bash
   python -m unittest test_functions.TestClass.test_sdedit -v
   ```
   3.2 [2 pts Manually Graded] SDEdit and text-guided editing of the test
   image, and of a photo or drawing of your own.

   3.3 [2 pts Autograded] Visual anagrams, `anagram_velocity`.
   ```bash
   python -m unittest test_functions.TestClass.test_anagram_velocity -v
   ```
   3.4 [4 pts Manually Graded] Your own visual anagram: choose a pair of
   prompts that surprises us.

   Once the tests pass on your computer, run the Q3 experiments on a GPU. On
   Delta AI, upload `flow_editing.py`, `Q3_flow_editing.ipynb` (with your
   prompts) and your own image to your MP folder on Delta AI, and run the whole notebook
   as a batch job from the login node:
   ```bash
   sbatch run_notebook.sbatch Q3_flow_editing.ipynb
   ```
   The figures are saved to `outputs/`: download them for your report (e.g.
   `scp -O "<login>@dtai-login.delta.ncsa.illinois.edu:/work/nvme/bfdu/<login>/mp5/outputs/q3_*.png" .`). `Q3_flow_editing_run.ipynb` is the notebook with all its
   outputs. The whole notebook takes about 1.5 minutes on Delta AI (about 5 minutes on an
   RTX 3060).

### Acknowledgments
Q3 follows Part A of [Berkeley's CS180 project
3](https://cal-cs180.github.io/fa26/hw/proj3-flow/parta.html), and uses
[PixNerd](https://github.com/MCG-NJU/PixNerd) (MIT license) by Wang et al.
The Q2 dataset is [AFHQv2](https://github.com/clovaai/stargan-v2) by Choi et
al.
