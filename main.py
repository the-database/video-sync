#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import datetime

from align_videos_by_soundtrack.align import SyncDetector
from align_videos_by_soundtrack.align_params import SyncDetectorSummarizerParams
from align_videos_by_soundtrack.customalign import CustomSyncDetector
from align_videos_by_soundtrack.utils import check_and_decode_filenames


def format_timedelta(seconds):
    if seconds > 0:
        return str(datetime.timedelta(seconds=seconds))
    else:
        return '-' + format_timedelta(-seconds)


def align_videos(horrible_filename, ohys_filename):

    horrible_base = os.path.splitext(horrible_filename)[0]
    ohys_base = os.path.splitext(ohys_filename)[0]

    file_specs = check_and_decode_filenames(
        [horrible_filename, ohys_filename], min_num_files=2)

    custom_summarizer_params = SyncDetectorSummarizerParams(fft_bin_size=1024, overlap=512, box_width=1,
                                                            maxes_per_box=7)

    print("(1/3) Determining initial video offset...")

    with CustomSyncDetector(params=custom_summarizer_params) as det:
        # first, find alignment at the start of the videos
        align_start = det.align(
            file_specs,
            duration=40,
        )

        pad_ohys = align_start[1]['pad'] - align_start[1]['trim']

        print(f'The start of {ohys_base} must be shifted by: {format_timedelta(pad_ohys)}\n')
        print(f'(2/3) Determining timestamp where sponsor screen starts in {ohys_base}...')

        # then, find the timestamp where the post OP sponsor screen starts
        # this will to be cut from the final audio output
        mismatch_start = det.find_mismatch(
            file_specs,
            duration=60 * 7,  # TODO don't require a duration; perform the mismatch search while building the boxes
            delay=pad_ohys
        )

        print(f'Sponsor screen in {ohys_base} starts at: {format_timedelta(mismatch_start)}\n')
        print(f'(3/3) Determining timestamp where sponsor screen ends in {ohys_base}...')

    with SyncDetector(params=SyncDetectorSummarizerParams()) as det:
        # finally, find the timestamp where the post OP sponsor screen ends
        mismatch_end_result = det.align(
            file_specs,
        )

        mismatch_length = mismatch_end_result[1]['trim'] + pad_ohys

        print(f'Sponsor screen in {ohys_base} ends at: '
              f'{format_timedelta(mismatch_start + mismatch_length)}\n')

    print(f'The range from {format_timedelta(mismatch_start)} to {format_timedelta(mismatch_start + mismatch_length)} '
          f'will be cut out from {ohys_base}\n')

    # ffmpeg commands
    print('--------------------------------------------------------------------------------\n')

    # extract ohys aac
    ohys_aac = f'{ohys_base}.aac'
    extract_aac_command = ['ffmpeg', '-y', '-i', ohys_filename,
                           '-vn', '-acodec', 'copy', '-avoid_negative_ts', '1', '-copytb', '1', ohys_aac]
    print(f"(1/5) Extracting audio from {ohys_base}:\n{' '.join(extract_aac_command)}\n")
    subprocess.call(extract_aac_command,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # extract ohys part A
    ohys_part_a_name = f'{ohys_base}-part-a.aac'
    part_a_command = ['ffmpeg', '-y', '-i', ohys_aac, '-t', str(mismatch_start),
                      '-vn', '-acodec', 'copy', '-avoid_negative_ts', '1', ohys_part_a_name]
    print(f"(2/5) Extracting part A audio (before sponsor screen) from {ohys_aac}:\n{' '.join(part_a_command)}\n")
    subprocess.call(part_a_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # extract ohys part B
    ohys_part_b_name = f'{ohys_base}-part-b.aac'
    part_b_command = ['ffmpeg', '-y', '-i', ohys_aac, '-ss', str(mismatch_start + mismatch_length),
                      '-vn', '-acodec', 'copy', '-avoid_negative_ts', '1', ohys_part_b_name]
    print(f"(3/5) Extracting part B audio (after sponsor screen) from {ohys_aac}:\n{' '.join(part_b_command)}\n")
    subprocess.call(part_b_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # concat ohys part A and B
    # create a temporary file with the list of files to concatenate
    filelist = 'filelist.txt'
    ohys_merged_name = f'{ohys_base}-merged.aac'

    with open(filelist, 'wb') as fp:
        fp.write(f"file '{ohys_part_a_name}'\n"
                 f"file '{ohys_part_b_name}'\n".encode())

    merge_command = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                     '-i', filelist, '-c', 'copy', ohys_merged_name]
    print(f"(4/5) Merging part A and part B audio, resulting in sponsor screen audio removed from {ohys_aac}:\n{' '.join(merge_command)}\n")
    subprocess.call(merge_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # create new video with horrible video + ohys merged audio
    # print(f'pad_ohys {pad_ohys}')
    # print(f'mismatch {format_timedelta(mismatch_start)} - '
    #       f'{format_timedelta(mismatch_start + mismatch_length)}')

    video_command = [
        'mkvmerge', '-o', f'{os.path.join(os.path.dirname(ohys_base), os.path.basename(horrible_base))}-ohys.mkv',
        '-A', horrible_filename,  # horrible mkv source: exclude audio
        '-y', f'0:{int(round(pad_ohys * 1000))}', ohys_merged_name  # ohys merged aac: shift by ohys pad
    ]
    print(f"(5/5) Creating final video:\n{' '.join(video_command)}\n")
    subprocess.call(video_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # cleanup
    os.remove(filelist)
    os.remove(ohys_merged_name)
    os.remove(ohys_part_a_name)
    os.remove(ohys_part_b_name)
    os.remove(ohys_aac)


def main(args=sys.argv):
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} <low bitrate (horrible) file> <high bitrate (ohys) file>')
        exit()

    horrible_filename = sys.argv[1]
    ohys_filename = sys.argv[2]

    # horrible_filename = './videos/[HorribleSubs] Ahiru no Sora - 32 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Ahiru no Sora - 32 (AT-X 1280x720 x264 AAC).mp4'
    # horrible_filename = './videos/[HorribleSubs] Ahiru no Sora - 33 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Ahiru no Sora - 33 (AT-X 1280x720 x264 AAC).mp4'  # mismatch_start = ~6:43.2
    # horrible_filename = './videos/[HorribleSubs] Fruits Basket S2 (2019) - 07 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Fruits Basket (2020) - 07 (TX 1280x720 x264 AAC).mp4'
    # horrible_filename = './videos/[HorribleSubs] Yesterday wo Utatte - 06 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Yesterday o Utatte - 06 (EX 1280x720 x264 AAC).mp4'
    # horrible_filename = './videos/[HorribleSubs] Yesterday wo Utatte - 08 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Yesterday o Utatte - 08 (EX 1280x720 x264 AAC).mp4'
    # horrible_filename = './videos/[HorribleSubs] Yesterday wo Utatte - 09 [1080p].mkv'
    # ohys_filename = './videos/[Ohys-Raws] Yesterday o Utatte - 09 (EX 1280x720 x264 AAC).mp4'
    # horrible_filename = './videos/[Asenshi] Brand New Animal (BNA) - 06 [C7A4A3AD].mkv'
    # ohys_filename = './videos/[Ohys-Raws] BNA - 06 (CX 1280x720 x264 AAC).mp4'

    align_videos(horrible_filename, ohys_filename)


if __name__ == "__main__":
    main()
