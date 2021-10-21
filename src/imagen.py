from datetime import datetime as dt
from pathlib import Path
from os import kill
from os.path import exists, getmtime
from signal import SIGKILL
import subprocess
import time

import requests
import streamlit as st
from torch.multiprocessing import Process, set_start_method
import yaml

from generate import generate, VQArgs


CHECKPOINT_PATH = Path("checkpoints")
CHECKPOINT_PATH.mkdir(exist_ok=True)

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def check_process(pid=None):
    if pid is None:
        return False
    try:
        kill(pid, 0)
    except OSError:
        return False
    else:
        return True


@st.cache(suppress_st_warning=True)
def download_vqgan_model(model: str) -> str:
    vqgan_config = None
    for filename, url in config["VQGAN_MODELS"][model].items():
        filepath = CHECKPOINT_PATH.joinpath(filename)
        if filepath.suffix == ".yaml":
            vqgan_config = str(filepath)
        if filepath.exists():
            continue
        bar = st.progress(0)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with filepath.open("wb") as f:
            for idx, data in enumerate(response.iter_content(block_size)):
                f.write(data)
                bar.progress(min(idx * block_size / total_size, 1.0))
        bar.empty()
    return vqgan_config


st.title("Image Generator")

image_loc = st.empty()
loading_txt = st.empty()
if "output" in st.session_state:
    image_loc.image(st.session_state["output"])
else:
    loading_txt.subheader("")
bar = st.empty()

with st.sidebar:
    vqgan_model = st.selectbox(
        label="Select VQGAN Model",
        index=6,
        options=config["VQGAN_MODELS"],
    )

    prompts = st.text_input(label="Text prompt for image generation")

    max_iterations = st.number_input(label="Max Iterations", min_value=0, max_value=5000, value=500, step=10)
    display_freq = st.number_input(label="Refresh Every N Iterations", min_value=0, max_value=max_iterations, value=min(50, max_iterations), step=10)
    size = st.number_input(label="Image Size", min_value=32, max_value=1024, value=256, step=1)

    args = VQArgs()
    generator = None

    col1, col2, col3, col4, col5 = st.columns(5)
    with col2:
        run = st.button("Run")
    with col4:
        stop = st.button("Stop")

    if run:
        vqgan_config = download_vqgan_model(model=vqgan_model)
        now = dt.utcnow().strftime("%Y%m%d_%H%M%S")
        output = OUTPUT_PATH.joinpath(f"{now}_image.png")
        args = VQArgs(
            prompts=prompts,
            vqgan_config=vqgan_config,
            output=str(output),
            max_iterations=max_iterations,
            display_freq=display_freq,
            size=size,
        )

        bar.progress(0)
        generator = Process(target=generate, args=(args,), daemon=True)
        generator.start()
        st.session_state["pid"] = generator.pid
        st.session_state["output"] = args.output

    if stop:
        print("Killing process...")
        try:
            kill(st.session_state["pid"], SIGKILL)
        except (KeyError, OSError):
            pass

    count = 0
    modified_time = 0
    step = 1.0 / (args.max_iterations // args.display_freq)
    while check_process(pid=st.session_state.get("pid")):
        loading_txt.subheader("Generating image...")
        time.sleep(1)
        if exists(args.output) and getmtime(args.output) > modified_time:
            try:
                image_loc.image(args.output)
            except:
                # avoid race condition while image is being written
                continue
            bar.progress(max(.01, min(1.0, count * step)))
            modified_time = getmtime(args.output)
            count += 1
    if count > 0:
        loading_txt.subheader("Image generation finished")
    bar.empty()
