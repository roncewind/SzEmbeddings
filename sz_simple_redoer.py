#! /usr/bin/env python3

import concurrent.futures

import argparse
import orjson
import logging
import traceback

import importlib
import sys
import os
import time
import random

from senzing import (
    SzConfig,
    SzConfigManager,
    SzEngine,
    SzEngineFlags,
    SzBadInputError,
)
from senzing.szerror import SzRetryableError, SzError

import senzing_core

INTERVAL = 1000
LONG_RECORD = os.getenv("LONG_RECORD", default=300)
EMPTY_PAUSE_TIME = int(os.getenv("SENZING_REDO_SLEEP_TIME_IN_SECONDS", default=60))

TUPLE_MSG = 0
TUPLE_STARTTIME = 1

log_format = "%(asctime)s %(message)s"

def loggingID(rec):
    dsrc = rec.get("DATA_SOURCE")
    rec_id = rec.get("RECORD_ID")
    if dsrc and rec_id:
        return f'{dsrc} : {rec_id}'
    umf_proc = rec.get("UMF_PROC") # repair messages
    if umf_proc:
        return 'REPAIR_ENTITY'
    return "UNKNOWN RECORD"

def process_msg(engine, msg, info):
    try:
        if info:
            response = engine.process_redo_record(msg, SzEngineFlags.SZ_WITH_INFO)
            return response
        else:
            engine.process_redo_record(msg)
            return None
    except Exception as err:
        print(f"{err} [{msg}]", file=sys.stderr)
        raise


try:
    log_level_map = {
        "notset": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "fatal": logging.FATAL,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level_parameter = os.getenv("SENZING_LOG_LEVEL", "info").lower()
    log_level = log_level_map.get(log_level_parameter, logging.INFO)
    logging.basicConfig(format=log_format, level=log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--info",
        dest="info",
        action="store_true",
        default=False,
        help="produce withinfo messages",
    )
    parser.add_argument(
        "-t",
        "--debugTrace",
        dest="debugTrace",
        action="store_true",
        default=False,
        help="output debug trace information",
    )
    args = parser.parse_args()

    engine_config = os.getenv("SENZING_ENGINE_CONFIGURATION_JSON")
    if not engine_config:
        print(
            "The environment variable SENZING_ENGINE_CONFIGURATION_JSON must be set with a proper JSON configuration.",
            file=sys.stderr,
        )
        print(
            "Please see https://senzing.zendesk.com/hc/en-us/articles/360038774134-G2Module-Configuration-and-the-Senzing-API",
            file=sys.stderr,
        )
        exit(-1)

    # Initialize Sz
    factory = senzing_core.SzAbstractFactoryCore("sz_simple_redoer", engine_config, verbose_logging=args.debugTrace)
    g2 = factory.create_engine()
    logCheckTime = prevTime = time.time()

    max_workers = int(os.getenv("SENZING_THREADS_PER_PROCESS", 0))

    if not max_workers:  # reset to null for executors
        max_workers = None

    messages = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        print(f"Threads: {executor._max_workers}")
        futures = {}
        empty_pause = 0
        try:
            while True:

                nowTime = time.time()
                if futures:
                    done, _ = concurrent.futures.wait(
                        futures,
                        timeout=10,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    delete_batch = []
                    delete_cnt = 0

                    for fut in done:
                        msg = futures.pop(fut)
                        try:
                            result = fut.result()
                            if result:
                                print(
                                    result
                                )  # we would handle pushing to withinfo queues here BUT that is likely a second future task/executor
                        except (SzRetryableError, SzBadInputError) as err:
                                record = orjson.loads(msg[TUPLE_MSG])
                                print(
                                      f'FAILED due to bad data or retryable error: {record["DATA_SOURCE"]} : {record["RECORD_ID"]}'
                                )
                        except SzError as err:
                                # Check for SENZ1001 "too long" errors which are also retryable
                                error_msg = str(err)
                                if 'too long' in error_msg.lower() or 'SENZ1001' in error_msg:
                                    record = orjson.loads(msg[TUPLE_MSG])
                                    print(
                                          f'FAILED due to SENZ1001 too long (retryable): {record["DATA_SOURCE"]} : {record["RECORD_ID"]}'
                                    )
                                else:
                                    # Re-raise other Senzing errors
                                    raise

                        messages += 1

                        if messages % INTERVAL == 0:  # display rate stats
                            diff = nowTime - prevTime
                            speed = -1
                            if diff > 0.0:
                                speed = int(INTERVAL / diff)
                            print(
                                f"Processed {messages} redo, {speed} records per second"
                            )
                            prevTime = nowTime

                    if nowTime > logCheckTime + (
                        LONG_RECORD / 2
                    ):  # log long running records
                        logCheckTime = nowTime

                        response = g2.get_stats()
                        print(f"\n{response}\n")

                        numStuck = 0
                        numRejected = 0
                        for fut, msg in futures.items():
                            if not fut.done():
                                duration = nowTime - msg[TUPLE_STARTTIME]
                                if duration > LONG_RECORD * 2:
                                    numStuck += 1
                                    record = orjson.loads(msg[TUPLE_MSG])
                                    print(
                                        f'Long record ({duration/60:.1f} min): {loggingID(record)}'
                                    )
                            if numStuck >= executor._max_workers:
                                print(
                                    f"All {executor._max_workers} threads are stuck on long running records"
                                )

                # switch to getDatasourceInfo
                #pauseSeconds = governor.govern()
                # either governor fully triggered or our executor is full
                # not going to get more messages
                #if pauseSeconds < 0.0:
                #    time.sleep(1)
                #    continue
                if len(futures) >= executor._max_workers:
                    time.sleep(1)
                    continue
                #if pauseSeconds > 0.0:
                #    time.sleep(pauseSeconds)

                if empty_pause:
                    if time.time() < empty_pause:
                        time.sleep(1)
                        continue
                    empty_pause = 0

                while len(futures) < executor._max_workers:
                    try:
                        response = g2.get_redo_record()
                        # print(response)
                        if not response:
                            print(
                                f"No redo records available. Pausing for {EMPTY_PAUSE_TIME} seconds."
                            )
                            empty_pause = time.time() + EMPTY_PAUSE_TIME
                            break
                        msg = response
                        futures[executor.submit(process_msg, g2, msg, args.info)] = (
                            msg,
                            time.time(),
                        )
                    except Exception as err:
                        print(f"{type(err).__name__}: {err}", file=sys.stderr)
                        raise

            print(f"Processed total of {messages} redo")

        except Exception as err:
            print(
                f"{type(err).__name__}: Shutting down due to error: {err}",
                file=sys.stderr,
            )
            traceback.print_exc()
            nowTime = time.time()
            for fut, msg in futures.items():
                if not fut.done():
                    duration = nowTime - msg[TUPLE_STARTTIME]
                    record = orjson.loads(msg[TUPLE_MSG])
                    print(
                        f'Still processing ({duration/60:.1f} min: {loggingID(record)}'
                    )
            executor.shutdown()
            exit(-1)

except Exception as err:
    print(err, file=sys.stderr)
    traceback.print_exc()
    exit(-1)
