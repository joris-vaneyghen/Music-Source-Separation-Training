import csv
import os
import sys


def write_csv(dirs, output_file, allowed_instruments=None):
    """Write a CSV file listing all instruments and their paths, and log total ms per instrument."""
    instrument_totals = {}

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instrum', 'path'])

        for instrument, path, duration_ms in find_audio_files(dirs, allowed_instruments):
            writer.writerow([instrument, path])

            # Accumulate duration per instrument
            if instrument in instrument_totals:
                instrument_totals[instrument] += duration_ms
            else:
                instrument_totals[instrument] = duration_ms

    # Log total time per instrument in hours and minutes
    print("Total time per instrument:")
    for instrument, total_ms in sorted(instrument_totals.items()):
        total_seconds = total_ms / 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        print(f"{instrument}: {int(hours)}h {int(minutes)}m")


def find_audio_files(dirs, allowed_instruments=None):
    """Recursively find all .wav and .flac files in the given directories and yield (instrument, path, duration_ms) tuples."""
    for directory in dirs:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    full_path = os.path.join(root, file)

                    # Parse instrument name and calculate duration
                    instrument, duration_ms = parse_audio_filename(file)

                    if instrument and duration_ms is not None:
                        # Filter by allowed instruments if specified
                        if allowed_instruments is None or instrument in allowed_instruments:
                            yield instrument, full_path, duration_ms


def parse_audio_filename(filename):
    """
    Parse filename with format: instrument#nb_start_end.flac
    Returns: (instrument, duration_ms) where duration_ms = end - start
    """
    try:
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]

        parts = name_without_ext.split('#')

        if len(parts) == 2:
            # The first part contains instrument#nb
            instrument = parts[0]
            meta_data = parts[1]
            meta_data_parts = meta_data.split('_')

            # Get start and end values (last two parts)
            start = int(meta_data_parts[-2])
            end = int(meta_data_parts[-1])

            duration_ms = end - start

            return instrument, duration_ms
        else:
            print(f"Warning: Unexpected filename format: {filename}", file=sys.stderr)
            return None, None

    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse filename {filename}: {e}", file=sys.stderr)
        return None, None


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process audio files and create CSV catalog')
    parser.add_argument('directories', nargs='+', help='Directories to search for audio files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--instruments', '-i', nargs='*',
                        help='List of allowed instruments to include (if not specified, all instruments are included)')

    args = parser.parse_args()

    directories_to_search = args.directories
    output_csv = args.output
    allowed_instruments = set(args.instruments) if args.instruments else None

    # Validate directories exist
    for directory in directories_to_search:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
            sys.exit(1)

    print(f"Searching directories: {directories_to_search}")
    if allowed_instruments:
        print(f"Filtering instruments: {allowed_instruments}")
    else:
        print("Including all instruments")
    print(f"Output file: {output_csv}")
    print("-" * 50)

    write_csv(directories_to_search, output_csv, allowed_instruments)