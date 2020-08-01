#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script based on alignment_by_row_channels.py by Allison Deal, see
# https://github.com/allisonnicoledeal/VideoSync/blob/master/alignment_by_row_channels.py
"""
This module contains the detector class for knowing the offset
difference for audio and video files, containing audio recordings
from the same event. It relies on ffmpeg being installed and
the python libraries scipy and numpy.
"""
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
from collections import defaultdict
import math
import json
import tempfile
import shutil
import logging
import datetime

import numpy as np

from . import communicate
from .utils import check_and_decode_filenames
from . import _cache
from . import cli_common
from .align_params import SyncDetectorSummarizerParams

__all__ = [
    'SyncDetectorSummarizerParams',
    'CustomSyncDetector',
    'main',
]

_logger = logging.getLogger(__name__)


class _FreqTransSummarizer(object):
    def __init__(self, working_dir, params):
        self._working_dir = working_dir
        self._params = params

    def _summarize(self, data, duration, ignore):
        """
        Return characteristic frequency transition's summary.
    
        The dictionaries to be returned are as follows:
        * key: The frequency appearing as a peak in any time zone.
        * value: A list of the times at which specific frequencies occurred.
        """
        freqs_dict = defaultdict(list)
        time_dict = defaultdict(list)

        boxes = defaultdict(list)

        last_sample = len(data)

        if duration is not None:
            last_sample = int(self._params.sample_rate * duration)

        start = int(-self._params.overlap)

        if ignore is not None:
            start += int(ignore * self._params.sample_rate)

        for x, j in enumerate(
                range(
                    start,
                    last_sample,
                    int(self._params.fft_bin_size - self._params.overlap))):

            sample_data = data[max(0, j):max(0, j) + self._params.fft_bin_size]
            # print(f"{x}, {j}, {max(0, j)} to {max(0, j) + self._params.fft_bin_size}, {self._x_to_secs(x)}")
            # if there are enough audio points left to create a full fft bin
            if len(sample_data) == self._params.fft_bin_size:
                intensities = np.abs(np.fft.fft(sample_data))  # intensities is list of fft results
                box_x = x // self._params.box_width
                for y in range(len(intensities) // 2):
                    box_y = y // self._params.box_height
                    # x: corresponding to time
                    # y: corresponding to freq
                    if self._params.lowcut is not None and \
                            isinstance(self._params.lowcut, (int,)):
                        if y <= self._params.lowcut:
                            continue
                    if self._params.highcut is not None and \
                            isinstance(self._params.highcut, (int,)):
                        if y >= self._params.highcut:
                            continue

                    if intensities[y] > 0:
                        boxes[(box_x, box_y)].append((intensities[y], x, y))

                        if len(boxes[(box_x, box_y)]) > self._params.maxes_per_box:
                            boxes[(box_x, box_y)].remove(min(boxes[(box_x, box_y)]))

        for box_x, box_y in list(boxes.keys()):
            for intensity, x, y in boxes[(box_x, box_y)]:
                freqs_dict[y].append(x)
                time_dict[x].append(y)
        del boxes

        return freqs_dict, time_dict, self._x_to_secs, self._secs_to_x

    def _secs_to_x(self, secs):
        j = secs * float(self._params.sample_rate)
        x = (j + self._params.overlap) / (self._params.fft_bin_size - self._params.overlap)
        return x

    def _x_to_secs(self, x):
        j = x * (self._params.fft_bin_size - self._params.overlap) - self._params.overlap
        return float(j) / self._params.sample_rate

    # test
    def _x_to_duration(self, x):
        j = x * (self._params.fft_bin_size - self._params.overlap)
        return float(j) / self._params.sample_rate

    def _y_to_Hz(self, y):
        lower = y * self._params.sample_rate / self._params.fft_bin_size
        upper = lower +  self._params.sample_rate / self._params.fft_bin_size
        return (lower, upper, )

    def _summarize_wav(self, wavfile, duration, ignore):
        raw_audio, rate = communicate.read_audio(wavfile)
        freq_result, time_result, _x_to_secs, _secs_to_x = self._summarize(raw_audio, duration, ignore)
        del raw_audio
        return rate, freq_result, time_result, _x_to_secs, _secs_to_x

    def _extract_audio(self, video_file, duration):
        """
        Extract audio from video file, save as wav audio file

        INPUT: Video file, and its index of input file list
        OUTPUT: Does not return any values, but saves audio as wav file
        """
        return communicate.media_to_mono_wave(
            video_file, self._working_dir,
            duration=duration,
            sample_rate=self._params.sample_rate,
            afilter=self._params.afilter)

    def summarize_audiotrack(self, media, duration, ignore):
        _logger.info("for '%s' begin", os.path.basename(media))
        exaud_args = dict(video_file=media, duration=self._params.max_misalignment)
        # First, try getting from cache.
        # for_cache = dict(exaud_args)
        # for_cache.update(self._params.__dict__)
        # for_cache.update(dict(
        #     atime=os.path.getatime(media)
        # ))
        # ck = _cache.make_cache_key(**for_cache)
        # cv = _cache.get("_align", ck)
        # if cv:
        #     _logger.info("for '%s' end", os.path.basename(media))
        #     return cv[1:]

        # Not found in cache.
        _logger.info("extracting audio tracks for '%s' begin", os.path.basename(media))
        wavfile = self._extract_audio(**exaud_args)
        _logger.info("extracting audio tracks for '%s' end", os.path.basename(media))
        rate, ft_dict, time_dict, _x_to_secs, _secs_to_x = self._summarize_wav(wavfile, duration, ignore)
        # _cache.set("_align", ck, (rate, ft_dict, time_dict, _x_to_secs))
        _logger.info("for '%s' end", os.path.basename(media))
        return ft_dict, time_dict, _x_to_secs, _secs_to_x

    def find_delay(
            self,
            freqs_dict_orig, freqs_dict_sample,
            min_delay=float('nan'),
            max_delay=float('nan')):
        #
        min_delay, max_delay = self._secs_to_x(min_delay), self._secs_to_x(max_delay)
        keys = set(freqs_dict_sample.keys()) & set(freqs_dict_orig.keys())
        #
        if not keys:
            raise Exception(
                """I could not find a match. Consider giving a large value to \
"max_misalignment" if the target medias are sure to shoot the same event.""")
        #
        if freqs_dict_orig == freqs_dict_sample:
            return 0.0
        #
        t_diffs = defaultdict(int)
        for key in keys:
            for x_i in freqs_dict_sample[key]:  # determine time offset
                for x_j in freqs_dict_orig[key]:
                    delta_t = x_i - x_j
                    mincond_ok = math.isnan(min_delay) or delta_t >= min_delay
                    maxcond_ok = math.isnan(max_delay) or delta_t <= max_delay
                    if mincond_ok and maxcond_ok:
                        t_diffs[delta_t] += 1
        try:
            return self._x_to_duration(
                sorted(list(t_diffs.items()), key=lambda x: -x[1])[0][0])
        except IndexError as e:
            raise Exception(
                """I could not find a match. \
Are the target medias sure to shoot the same event?""")


class CustomSyncDetector(object):
    def __init__(self, params=SyncDetectorSummarizerParams(), clear_cache=False):
        self._working_dir = tempfile.mkdtemp()
        self._impl = _FreqTransSummarizer(
            self._working_dir, params)
        self._orig_infos = {}  # per filename
        if clear_cache:
            _cache.clean("_align")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        retry = 3
        while retry > 0:
            try:
                shutil.rmtree(self._working_dir)
                break
            except:
                import time
                retry -= 1
                time.sleep(1)

    def _get_media_info(self, fn):
        if fn not in self._orig_infos:
            self._orig_infos[fn] = communicate.get_media_info(fn)
        return self._orig_infos[fn]

    def _isclose(self, freqs_a, freqs_b):

        # max_delta = 1
        #
        # deltas = []
        #
        # for a in freqs_a:
        #     for b in freqs_b:
        #         if abs(a - b) < max_delta:
        #             return True
        # return False
        return len(set(freqs_a) & set(freqs_b)) >= len(freqs_a) // 2

    def _find_ohys_in_horrible(self, x_ohys, x_horrible, horrible, ohys):
        # window = 2
        # x_js = range(max(0, x_horrible - window), x_horrible + window)
        # for x_horrible in x_js:
        if x_horrible in horrible:
            if self._isclose(horrible[x_horrible], ohys[x_ohys]):
                return True
        # print('hard miss')
        return False

    def _find_horrible_in_ohys(self, x_horrible, x_ohys, ohys, horrible):
        return self._find_ohys_in_horrible(x_horrible, x_ohys, ohys, horrible)

    def _find_mismatch(self, ftds, delay, _x_to_secs, _secs_to_x):
        miss_requirement = 15  # number of ohys misses allowed before it is skipped/ignored
        hit_tolerance = 3  # number of hits needed to reset miss counter
        miss_length = 0
        hit_length = 0
        miss_start = None
        # x_offset = int(_secs_to_x(delay))
        x_offset = int(delay * 48000 / 512)
        # x_offset = -(1.05 * 48000)
        # x_offset = 99
        horrible = ftds[0]
        ohys = ftds[1]

        miss_start_candidates = {}

        for x_ohys in ohys:
            if x_ohys > 1000:  # test skip first 10 seconds
                x_horrible = x_ohys + x_offset
                ohys_chunk_is_present = self._find_ohys_in_horrible(x_ohys, x_horrible, horrible, ohys)

                if ohys_chunk_is_present:
                    hit_length += 1
                    # print(f"HIT at x = horrible={x_horrible},{datetime.timedelta(seconds=_x_to_secs(x_horrible))} {horrible[x_horrible]}\t\t\t\tohys={x_ohys},{datetime.timedelta(seconds=_x_to_secs(x_ohys))} {ohys[x_ohys]}")

                    if hit_length >= hit_tolerance:
                        # reset miss counters
                        miss_start = None
                        miss_length = 0
                else:
                    # print(f'miss at x = horrible={x_horrible},{datetime.timedelta(seconds=_x_to_secs(x_horrible))} {horrible[x_horrible]}\t\tohys={x_ohys},{datetime.timedelta(seconds=_x_to_secs(x_ohys))}\t\tmiss_length={miss_length} {ohys[x_ohys]}')
                    if miss_start is None:
                        miss_start = x_ohys
                    miss_length += 1
                    hit_length = 0

                if miss_length > miss_requirement:
                    # miss_start_candidates[miss_start] = miss_length
                    break

                # if len(miss_start_candidates) >= 5:
                #     break

        # ohys post OP sponsor screen start
        return _x_to_secs(miss_start)

        # print(f'final missed at ohys_x = {miss_start},{datetime.timedelta(seconds=_x_to_secs(miss_start))}')
        #
        # hit_requirement = 60  # number of hits required before accepting a match
        # miss_tolerance = 3  # number of misses needed to reset hit counter
        # hit_length = 0
        # miss_length = 0
        #
        # miss_horrible_start = 0
        #
        # for x_horrible in horrible:
        #     if x_horrible > miss_start + x_offset:  # pad 300 test
        #         miss_horrible_start = x_horrible
        #         break
        #
        # x_ohys = 0
        # x_offset *= -1
        # x_horrible = miss_horrible_start
        # hit_start = -1
        #
        # while x_ohys < max(list(ohys.keys())):
        #     if x_horrible in horrible:
        #         x_ohys = x_horrible + x_offset
        #
        #         horrible_chunk_is_present = self._find_ohys_in_horrible(x_horrible, x_ohys, ohys, horrible)
        #
        #         if horrible_chunk_is_present:
        #             hit_length += 1
        #             print(f"HIT at horrible={x_horrible},{datetime.timedelta(seconds=_x_to_secs(x_horrible))}\t\t\t\tohys={x_ohys},{datetime.timedelta(seconds=_x_to_secs(x_ohys))}\t\toffset={x_offset},{datetime.timedelta(seconds=(_x_to_secs(x_offset)))}\t\thit_length={hit_length}")
        #             if hit_start == -1:
        #                 hit_start = x_ohys
        #             miss_length = 0
        #             x_horrible += 1
        #         else:
        #             miss_length += 1
        #             print(f'soft miss at x = horrible={x_horrible},{datetime.timedelta(seconds=_x_to_secs(x_horrible))}\t\tohys={x_ohys},{datetime.timedelta(seconds=_x_to_secs(x_ohys))}\t\toffset={x_offset},{datetime.timedelta(seconds=(_x_to_secs(x_offset)))}')
        #             if miss_length >= miss_tolerance:
        #                 print(f'missed at x = horrible={x_horrible},{datetime.timedelta(seconds=_x_to_secs(x_horrible))}\t\t\tohys={x_ohys},{datetime.timedelta(seconds=_x_to_secs(x_ohys))}\t\toffset={x_offset},{datetime.timedelta(seconds=(_x_to_secs(x_offset)))}')
        #                 x_horrible = miss_horrible_start
        #                 hit_length = 0
        #                 x_offset += 1
        #                 hit_start = -1
        #                 miss_length = 0
        #             else:
        #                 x_horrible += 1
        #         if hit_length > hit_requirement:
        #             break
        #     else:
        #         x_horrible += 1
        #
        # # print('target guess', _x_to_secs(6127), _x_to_secs(6127) - _x_to_secs(miss_start[0]))
        # print(f'final missed at x = {miss_start}, {hit_start}, {datetime.timedelta(seconds=_x_to_secs(miss_start))}, '
        #       f'{datetime.timedelta(seconds=_x_to_secs(hit_start))}, {_x_to_secs(hit_start) - _x_to_secs(miss_start)}')
        # return miss_start, hit_start

    def _align(self, files, known_delay_map, duration, delay):
        """
        Find time delays between video files
        """

        def _each(idx, delay=None, ignore=None):
            return self._impl.summarize_audiotrack(files[idx], duration, ignore)

        tmp_ftds = {i: _each(i) for i in range(len(files))}

        # tmp_ftds = {
        #     0: _each(0),
        #     1: _each(1, delay)
        # }

        # time_ftds = {i: tmp_ftds[i][1] for i in tmp_ftds}

        ftds = {i: tmp_ftds[i][0] for i in tmp_ftds}

        # for i in range(len(files)):
        #     with open(files[i] + '.txt', 'w') as f:
        #         for key in tmp_ftds[i][1]:
        #             f.write(f"{key : 010d} {datetime.timedelta(seconds=tmp_ftds[0][2](key))}: {tmp_ftds[i][1][key].__repr__()}\n")

        # if delay is not None:
        #     self._find_mismatch(time_ftds, delay, tmp_ftds[0][2], tmp_ftds[0][3])
        #     return  # TODO

        _result1, _result2 = {}, {}
        for kdm_key in known_delay_map.keys():
            kdm = known_delay_map[kdm_key]
            ft = os.path.abspath(kdm_key)
            fb = os.path.abspath(kdm["base"])
            it_all = [i for i, f in enumerate(files) if f == ft]
            ib_all = [i for i, f in enumerate(files) if f == fb]
            for it in it_all:
                for ib in ib_all:
                    _result1[(ib, it)] = -self._impl.find_delay(
                        ftds[ib], ftds[it],
                        kdm.get("min", float('nan')), kdm.get("max", float('nan')))
        #
        _result2[(0, 0)] = 0.0
        for i in range(len(files) - 1):
            if (0, i + 1) in _result1:
                _result2[(0, i + 1)] = _result1[(0, i + 1)]
            elif (i + 1, 0) in _result1:
                _result2[(0, i + 1)] = -_result1[(i + 1, 0)]
            else:
                _result2[(0, i + 1)] = -self._impl.find_delay(ftds[0], ftds[i + 1])
        #        [0, 1], [0, 2], [0, 3]
        # known: [1, 2]
        # _______________^^^^^^[0, 2] must be calculated by [0, 1], and [1, 2]
        # 
        # known: [1, 2], [2, 3]
        # _______________^^^^^^[0, 2] must be calculated by [0, 1], and [1, 2]
        # _______________^^^^^^^^[0, 3] must be calculated by [0, 2], and [2, 3]
        for ib, it in sorted(_result1.keys()):
            for i in range(len(files) - 1):
                if it == i + 1 and (0, i + 1) not in _result1 and (i + 1, 0) not in _result1:
                    if files[0] != files[it]:
                        _result2[(0, it)] = _result2[(0, ib)] - _result1[(ib, it)]
                elif ib == i + 1 and (0, i + 1) not in _result1 and (i + 1, 0) not in _result1:
                    if files[0] != files[ib]:
                        _result2[(0, ib)] = _result2[(0, it)] + _result1[(ib, it)]

        # build result
        result = np.array([_result2[k] for k in sorted(_result2.keys())])
        pad_pre = result - result.min()
        _logger.debug(
            list(sorted(zip(
                map(os.path.basename, files),
                [communicate.duration_to_hhmmss(pp) for pp in pad_pre]))))  #
        trim_pre = -(pad_pre - pad_pre.max())
        #
        return pad_pre, trim_pre

    def get_media_info(self, files):
        """
        Get information about the media (by calling ffprobe).

        Originally the "align" method had been internally acquired to get
        "pad_post" etc. When trying to implement editing processing of a
        real movie, it is very frequent to want to know these information
        (especially duration) in advance. Therefore we decided to release
        this as a method of this class. Since the retrieved result is held
        in the instance variable of class, there is no need to worry about
        performance.
        """
        files = check_and_decode_filenames(files)
        return [self._get_media_info(fn) for fn in files]

    def find_mismatch(self, files, duration=None, delay=None):

        def _each(idx, delay=None, ignore=None):
            return self._impl.summarize_audiotrack(files[idx], duration, ignore)

        tmp_ftds = {i: _each(i) for i in range(len(files))}
        time_ftds = {i: tmp_ftds[i][1] for i in tmp_ftds}

        ftds = {i: tmp_ftds[i][0] for i in tmp_ftds}

        return self._find_mismatch(time_ftds, delay, tmp_ftds[0][2], tmp_ftds[0][3])

    def align(
            self, files, known_delay_map={}, duration=None, delay=None):
        """
        Find time delays between video files
        """
        files = check_and_decode_filenames(files)
        pad_pre, trim_pre = self._align(
            files, known_delay_map, duration, delay)
        #
        infos = self.get_media_info(files)

        orig_dur = np.array([inf["duration"] for inf in infos])

        strms_info = [
            (inf["streams"], inf["streams_summary"]) for inf in infos]
        pad_post = list(
            (pad_pre + orig_dur).max() - (pad_pre + orig_dur))
        trim_post = list(
            (orig_dur - trim_pre) - (orig_dur - trim_pre).min())
        #
        return [{
            "trim": trim_pre[i],
            "pad": pad_pre[i],
            "orig_duration": orig_dur[i],
            "trim_post": trim_post[i],
            "pad_post": pad_post[i],
            "orig_streams": strms_info[min(i, 1)][0],
            "orig_streams_summary": strms_info[min(i, 1)][1],
        }
            for i in range(len(infos))]

    @staticmethod
    def summarize_stream_infos(result_from_align):
        """
        This is a service function that calculates several summaries on
        information about streams of all medias returned by
        SyncDetector#align.

        Even if "align" has only detectable delay information, you are
        often in trouble. This is because editing for lineup of targeted
        plural media involves unification of sampling rates (etc) in many
        cases.

        Therefore, this function calculates the maximum sampling rate etc.
        through all files, and returns it in a dictionary format.
        """
        result = dict(
            max_width=0,
            max_height=0,
            max_sample_rate=0,
            max_fps=0.0,
            has_video=[],
            has_audio=[])
        for ares in result_from_align:
            summary = ares["orig_streams_summary"]  # per single media

            result["max_width"] = max(
                result["max_width"], summary["max_resol_width"])
            result["max_height"] = max(
                result["max_height"], summary["max_resol_height"])
            result["max_sample_rate"] = max(
                result["max_sample_rate"], summary["max_sample_rate"])
            result["max_fps"] = max(
                result["max_fps"], summary["max_fps"])

            result["has_video"].append(
                summary["num_video_streams"] > 0)
            result["has_audio"].append(
                summary["num_audio_streams"] > 0)
        return result


def _bailout(parser):
    parser.print_help()
    sys.exit(1)


def main(args=sys.argv):
    parser = cli_common.AvstArgumentParser(description="""\
This program reports the offset difference for audio and video files,
containing audio recordings from the same event. It relies on ffmpeg being
installed and the python libraries scipy and numpy.
""")
    parser.add_argument(
        '--json',
        action="store_true",
        help='To report in json format.', )
    parser.add_argument(
        'file_names',
        nargs="+",
        help='Media files including audio streams. \
It is possible to pass any media that ffmpeg can handle.', )
    parser.add_argument(
        '--duration',
        type=float,
        help='Limit the scan to the first X seconds of each video'
    )
    parser.add_argument(
        '--delay',
        type=float,
        help='Shift video by X seconds before analyzing'
    )
    parser.add_argument(
        '--ignore',
        type=float,
        help='Ignore first X seconds of video for comparison'
    )
    args = parser.parse_args(args[1:])
    known_delay_map = args.known_delay_map

    cli_common.logger_config()
    duration = args.duration
    delay = args.delay

    file_specs = check_and_decode_filenames(
        args.file_names, min_num_files=2)
    if not file_specs:
        _bailout(parser)
    with SyncDetector(
            params=args.summarizer_params,
            clear_cache=args.clear_cache) as det:
        result = det.align(
            file_specs,
            known_delay_map=known_delay_map,
            duration=duration,
            delay=delay
        )
    if args.json:
        # print(json.dumps(
        #     {'edit_list': list(zip(file_specs, result))}, indent=4, sort_keys=True))
        return result
    else:
        report = []
        for i, path in enumerate(file_specs):
            # if not (result[i]["trim"] > 0):
            #     continue
            report.append(
                """Result: The beginning of '%s' needs to be trimmed off %.4f seconds \
(or to be added %.4f seconds padding) for all files to be in sync""" % (
                    path, result[i]["trim"], result[i]["pad"]))
        if report:
            print("\n".join(report))
        else:
            print("files are in sync already")


if __name__ == "__main__":
    main()
