# RUNBOOK G2: Kaggle feasibility check (for Thông)

Goal: measure whether GPT-2 XL + LoRA fine-tuning fits on one free Kaggle GPU and how long the full run
would take. You run one notebook cell, then copy the summary back to the lead. Nothing is uploaded by an
agent and no token is needed (the GPT-2 weights on Hugging Face are public).

Expected time: about 20-35 min of GPU time (XL download about 6.4 GB, then ~16 short timed runs). If XL
misses the 5 h budget, the same cell automatically goes on to GPT-2 Large (and then Medium), which adds
about 10-15 min.

## 0. Before you start (once)

- Kaggle account with a **verified phone number** (Kaggle -> your avatar -> Settings -> Phone verification).
  Without it the notebook cannot use a GPU or the Internet.
- The lead has run `python kaggle/make_bundle.py`, so `llm_ver2/kaggle/feasibility_bundle.py` is up to date
  (the header line says "GENERATED ... on <date time>").

## 1. Create the notebook

1. Go to https://www.kaggle.com/code and click **+ New Notebook**.
2. In the right-hand panel open **Session options** (or the menu **Settings**):
   - **Accelerator**: choose **GPU T4 x2**. (Only if T4 is not offered, choose **GPU P100**.)
   - **Internet**: switch **On**.
   - Leave Language = Python, Environment = "Pin to original" / default.
3. Wait until the session shows as running (green dot / "Session started"). Accept the GPU dialog if asked.

## 2. Paste and run

1. On your PC open `D:\School Courses\Sem4\Concept Innovative\assignment-2\llm_ver2\kaggle\feasibility_bundle.py`
   in VS Code or Notepad, press **Ctrl+A**, then **Ctrl+C**.
2. In the notebook click into the first code cell, delete what is there, and press **Ctrl+V**.
   (One cell holds the whole file; it is about 1,050 lines.)
3. Press **Shift+Enter** (or the play button) to run the cell.
4. What you should see, in order:
   - `[setup] pip install tiktoken` (maybe) and `[setup] wrote 5 module files to /tmp/fraudlm_src`
   - `[env] ...` and `[gpu] 2 GPU(s); using Tesla T4, 14.7 GB ...`
   - `[weights] downloading openai-community/gpt2-xl/model.safetensors ...` with a progress bar (1-5 min).
     A warning about `HF_TOKEN` not being set is normal; ignore it.
   - `[sanity] fp32-base: loss on sample text ... | block output dtype ... | greedy: 'Every effort moves you ...'`
     (loss should be roughly 2-4 and the text should be readable English). The fp32-base timings follow.
   - Then the fp16 base is loaded and checked against fp32:
     `[fp16 check] gpt2-xl: fp32 base 3.xxxx vs fp16 base 3.xxxx -> |diff| 0.00xx (limit 0.05) -> fp16 base ACCEPTED`.
     If it says `REJECTED`, the fp16-base timings are skipped. That is a result too; the lead then uses fp32 base.
   - One line per setting, for example
     `- fp16-base ckpt=on  mb=2 seq= 512 ... 0.800 s/step, 1,280 tok/s, peak 5.10 GB, ..., projected 1.92 h`
     Lines that say `OOM` are expected for the biggest settings; that is part of the measurement.
   - At the end a block that starts with `G2 FEASIBILITY SUMMARY` and ends with `[saved] /kaggle/working/g2_feasibility_results.json`.
5. Do not close the browser tab while it runs (a closed tab can stop an interactive session).

## 3. What to send back to the lead

- Copy **everything from the line `G2 FEASIBILITY SUMMARY` to the end of the output** and paste it into the
  project discussion. That block has the GPU name, the full results table and the `RECOMMENDED:` line.
- If anything failed, copy the **last 40 lines** of the output instead (the error message is at the bottom).
- Optional: in the right panel, **Output** -> `/kaggle/working/g2_feasibility_results.json` -> download, and
  put it in `llm_ver2/kaggle/results/` on your PC.

## 4. Stop the session

Click the **power / Stop session** button (top right) when the summary is printed. A running GPU session
uses your weekly GPU quota (30 h) even when idle. The weights are in `/tmp`, so they are not saved and do
not count toward the notebook output size.

## 5. If something goes wrong

| What you see | Do this |
|---|---|
| `CUDA is not available` | Accelerator is still "None". Set GPU T4 x2 (step 1.2), then run the cell again. |
| `this PyTorch build cannot run on Tesla P100` or `no kernel image is available` | Switch Accelerator to **GPU T4 x2** and run again. |
| `ConnectionError`, `Max retries exceeded`, `Temporary failure in name resolution` | Internet is Off (step 1.2) or your phone is not verified. Turn it on and run again. |
| The cell stops with "Your notebook tried to allocate more memory than is available" (CPU RAM) | Send the last 40 lines to the lead. |
| It is slow and you need to stop early | Click the stop (square) button. Send what is printed so far; each line is already a result. |
| You want a shorter run | In the cell, section "1. Settings", change `MODES = ["fp16-base", "fp32-base"]` to `MODES = ["fp16-base"]` and run again (about half the time). |

Never paste a Kaggle or Hugging Face token into the notebook or the chat; this check does not need one.
