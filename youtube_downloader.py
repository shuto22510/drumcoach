"""
YouTube Audio Downloader for AI Music Coach.
Downloads audio from YouTube videos for practice/research purposes.

NOTE: This is for personal research and learning only.
Do not use for commercial purposes or redistribute downloaded content.
"""

import sys
import os
from pathlib import Path

import yt_dlp


def download_audio(url: str, output_dir: str = "data", filename: str = None) -> str:
    """
    Download audio from YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        filename: Optional custom filename (without extension)

    Returns:
        Path to the downloaded audio file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path / (filename if filename else '%(title)s')) + '.%(ext)s',
        'quiet': False,
        'no_warnings': False,
    }

    print(f"\nDownloading audio from: {url}")
    print(f"Output directory: {output_path.absolute()}")
    print("-" * 60)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

            print(f"Title: {title}")
            print(f"Duration: {duration // 60}:{duration % 60:02d}")
            print("-" * 60)

            # Download
            ydl.download([url])

            # Determine output filename
            if filename:
                output_file = output_path / f"{filename}.wav"
            else:
                # Sanitize title for filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                output_file = output_path / f"{safe_title}.wav"

            print(f"\nDownload complete!")
            print(f"Saved to: {output_file}")

            return str(output_file)

    except Exception as e:
        print(f"\nError downloading: {e}")
        return None


def download_for_practice(url: str) -> None:
    """
    Download audio for practice session.
    Saves as 'student.wav' in data directory.

    Args:
        url: YouTube video URL
    """
    result = download_audio(url, output_dir="data", filename="student")
    if result:
        print("\nReady for practice!")
        print("Run: poetry run python main.py")


def main():
    if len(sys.argv) < 2:
        print("YouTube Audio Downloader for AI Music Coach")
        print("=" * 60)
        print("\nUsage:")
        print("  python youtube_downloader.py <youtube_url>")
        print("  python youtube_downloader.py <youtube_url> --name <filename>")
        print("  python youtube_downloader.py <youtube_url> --practice")
        print()
        print("Examples:")
        print("  # Download with original title")
        print("  python youtube_downloader.py https://www.youtube.com/watch?v=xxxxx")
        print()
        print("  # Download with custom filename")
        print("  python youtube_downloader.py https://www.youtube.com/watch?v=xxxxx --name my_song")
        print()
        print("  # Download as student.wav for immediate practice")
        print("  python youtube_downloader.py https://www.youtube.com/watch?v=xxxxx --practice")
        print()
        print("NOTE: For personal research/learning only. Do not redistribute.")
        return

    url = sys.argv[1]

    # Check for options
    if "--practice" in sys.argv:
        download_for_practice(url)
    elif "--name" in sys.argv:
        name_idx = sys.argv.index("--name")
        if name_idx + 1 < len(sys.argv):
            filename = sys.argv[name_idx + 1]
            download_audio(url, filename=filename)
        else:
            print("Error: --name requires a filename argument")
    else:
        download_audio(url)


if __name__ == "__main__":
    main()
