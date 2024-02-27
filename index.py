import sys
import yaml
import subprocess

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if not config['index']:
    raise Exception('config.yaml index vacío!')

for item in config['index']:
    url = item['url']
    lang = item['lang']
    cmd = f"yt-supercut index {url} --lang {lang}"
    subprocess.run(cmd, shell=True, check=True)
