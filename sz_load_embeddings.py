# -----------------------------------------------------------------------------
# Read Senzing JSONL file and import records with embeddings into Senzing.
# Example usage:
# python sz_load_embeddings.py -i /data/OpenSactions/senzing.json -o /data/OpenSactions/senzing.jsonl --name_model_path output/20250814/FINAL-fine_tuned_model  --biz_model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model
#

# -----------------------------------------------------------------------------
import argparse
import bz2
import concurrent.futures
import gzip
import itertools
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import numpy.typing as npt
import orjson
import torch
from sentence_transformers import SentenceTransformer
from senzing import SzEngine
from senzing_core import SzAbstractFactoryCore

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

file_path = ""
debug_on = True
number_of_lines_to_process = 0
number_of_names_to_process = 0
status_print_lines = 100


# =============================================================================
def debug(text):
    if debug_on:
        print(text, file=sys.stderr, flush=True)


# =============================================================================
def format_seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# =============================================================================
# create embedding for all the names in the list
def get_embeddings(names, model, batch_size) -> npt.NDArray[np.float16]:
    embeddings = model.encode(
        names,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float16, copy=False)
    return embeddings


# =============================================================================
# process json line and add to senzing
def process_json_line(line, name_model, biz_model, batch_size, engine: SzEngine):
    # Parse the JSON string
    entity = orjson.loads(line)
    record_type = entity["RECORD_TYPE"]

    model = None
    name_field = ""
    embed_field = ""
    label_field = ""
    if record_type == "PERSON":
        name_field = "NAME_FULL"
        embed_field = "NAME_EMBEDDING"
        label_field = "NAME_LABEL"
        model = name_model
    elif record_type == "ORGANIZATION":
        name_field = "NAME_ORG"
        embed_field = "BIZNAME_EMBEDDING"
        label_field = "BIZNAME_LABEL"
        model = biz_model

    names = entity["NAMES"]
    if names and model:
        # print(f'embed record: {entity.get("RECORD_ID", "")}')
        embed_list = []
        n_list = [n[name_field] for n in names]
        # embed singly
        e_list = [(n, get_embeddings(n, model, batch_size)) for n in n_list]
        # embed as batch
        # e_list = get_embeddings(n_list, model, batch_size)
        # debug(f"n_list: {len(n_list)}  e_list: {len(e_list)}")
        # for n, e in zip(n_list, e_list):
        #     debug(f"{n}: {e.tolist()}")
        embed_list = [{label_field: e[0], embed_field: f"{e[1].tolist()}"} for e in e_list]
        entity[embed_field + 'S'] = embed_list
    # debug(f'add_record: {entity.get("RECORD_ID", "")} : {json.dumps(entity, ensure_ascii=False)}')
    try:
        engine.add_record(entity.get("DATA_SOURCE", ""), entity.get("RECORD_ID", ""), json.dumps(entity, ensure_ascii=False))
    except Exception as err:
        debug(f"{err} [{json.dumps(entity, ensure_ascii=False)}]")
        raise
    return


# =============================================================================
# process a line from the file
def process_line(line, name_model, biz_model, batch_size, engine: SzEngine):
    stripped_line = line.strip()
    if "" == stripped_line or len(stripped_line) < 10:
        return
    # strip off the comma
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return process_json_line(stripped_line, name_model, biz_model, batch_size, engine)


# =============================================================================
# Read a jsonl file and process concurrently
def read_file_futures(file_handle, name_model, biz_model, batch_size, engine: SzEngine):
    line_count = 0
    shutdown = False
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_line, line, name_model, biz_model, batch_size, engine): line
            for line in itertools.islice(file_handle, executor._max_workers * 10)
        }
        print(f"📌 Threads: {executor._max_workers}")
        print("\n")
        print("☺", flush=True, end='\r')

        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for f in done:
                try:
                    f.result()
                    line_count += 1
                except Exception as e:
                    print(e)
                    pass
                else:
                    if not shutdown:
                        line = file_handle.readline()
                        if line:
                            futures[executor.submit(process_line, line, name_model, biz_model, batch_size, engine)] = (line)
                            # line_count += 1
                        if line_count % 1000 == 0:
                            debug(f"{engine.get_stats()}")
                        if line_count % status_print_lines == 0:
                            print(f"☺ {line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')
                        if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
                            executor.shutdown(wait=True, cancel_futures=False)
                            shutdown = True
                finally:
                    del futures[f]
    print("\n")
    print(f"{line_count:,} total lines read", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Read a jsonl file and process
def read_file(file_handle, name_model, biz_model, batch_size, engine: SzEngine):
    line_count = 0
    start_time = time.time()
    for line in file_handle:
        line_count += 1
        try:
            process_line(line, name_model, biz_model, batch_size, engine)
        except Exception as e:
            print(e)
            pass
        if line_count % status_print_lines == 0:
            print(f"☺ {line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')
        if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
            break
    print("\n")
    print(f"{line_count:,} total lines read", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_load_embeddings", description="Loads business name and personal name embeddings."
    )

    parser.add_argument("-i", "--infile", action="store", required=True, help='Path to Senzing JSON file.')
    parser.add_argument('--name_model_path', type=str, required=True, help='Path to personal names model.')
    parser.add_argument('--biz_model_path', type=str, required=True, help='Path to business names model.')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    args = parser.parse_args()

    infile_path = args.infile

    # check for CUDA
    device = "cpu"
    batch_size = 256
    if torch.cuda.is_available():
        batch_size = 256 if not args.batch_size else args.batch_size
        device = "cuda"
    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")

    print("⏳ Loading models...")
    name_model = SentenceTransformer(args.name_model_path)
    biz_model = SentenceTransformer(args.biz_model_path)

    print("⏳ Get Senzing engine...")
    settings = os.getenv("SENZING_ENGINE_CONFIGURATION_JSON")
    # print(settings)
    sz_factory = SzAbstractFactoryCore("SzEmbeddingsLoader", settings)
    sz_engine = sz_factory.create_engine()

    line_count = 0
    start_time = datetime.now()

    try:
        if infile_path.endswith(".bz2"):
            debug(f"Opening {infile_path}...")
            with bz2.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, name_model, biz_model, batch_size, sz_engine)
        elif infile_path.endswith(".gz"):
            debug(f"Opening {infile_path}...")
            with gzip.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, name_model, biz_model, batch_size, sz_engine)
        elif infile_path.endswith(".json") or infile_path.endswith(".jsonl"):
            debug(f"Opening {infile_path}...")
            with open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, name_model, biz_model, batch_size, sz_engine)
        elif infile_path == "-":
            debug("Opening stdin...")
            with open(sys.stdin.fileno(), 'rt') as f:
                line_count = read_file_futures(f, name_model, biz_model, batch_size, sz_engine)
        else:
            debug("Unrecognized file type.")
    except Exception:
        traceback.print_exc()

    end_time = datetime.now()
    print(f"Input read from {infile_path}.", flush=True)
    print(f"    Started at: {start_time}, ended at: {end_time}.", flush=True)
    print("\n-----------------------------------------\n")

# python sz_load_embeddings.py -i /data/OpenSactions/senzing.json -o /data/OpenSactions/senzing.jsonl --name_model_path output/20250814/FINAL-fine_tuned_model  --biz_model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model  --batch_size 1024 2> os-20250826.err
# python sz_load_embeddings.py -i s-test.json -o s-test.out --name_model_path output/20250814/FINAL-fine_tuned_model  --biz_model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model  --batch_size 1024 2> s-test.err

# python -c "from senzing_core import SzAbstractFactoryCore; f=SzAbstractFactoryCore('foo', '{}');print(f.create_product().get_version())"
