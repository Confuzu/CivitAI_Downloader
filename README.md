# CivitAI_Downloader
Downloader for Models /Loras/ Embedings for Direct API Urls 

small  helper when you get a txt file with direct API urls for downloading models from civitAI in this format

embedings
name.pt - https://civitai.com/api/download/models/ID

Lora
name.safetensors - https://civitai.com/api/download/models/ID

Model
name.safetensors - https://civitai.com/api/download/models/ID

Organizing them into appropriate folders and downloads only the files that are not already present in the specified folder.

## Authentication

Set your CivitAI API token as an environment variable (recommended):

```bash
export CIVITAI_API_TOKEN=your_token_here
```

If the variable is not set, you will be prompted to enter it at startup (input is hidden). The token is sent via `Authorization: Bearer` header — never appended to URLs.

## Usage

```bash
python civitAI_downloader.py --url_file my_urls.txt
```
Also includes a Retry mechanism for failed downloads
 ```
--retries 3
``` 
"Path to the file containing direct download URLs."
```
--url_file *(required)*
```
 "Maximum number of concurrent download threads." Default is 5
```
--max_threads
```
