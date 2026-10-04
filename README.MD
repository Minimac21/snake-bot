# Snake.io bot

## Install

Requirements:
- [tesseract-ocr](https://tesseractocr.org/install/)
- python

Installation:
- go to repo directory (folder)
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## Running the bot

`python bot.py`

-d: debug. saves per-frame info as jpgs

-t: time. prints per-tick timing information

## Scripts

start.sh

- removes any existing debug info and runs the bot

view.sh

- uses sxiv to view the debug info in order


## Status

I'm currently abandoning this project because it proved too difficult. I'm going to work on a slither.io bot, because it exposes state information.
