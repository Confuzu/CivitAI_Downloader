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
 Also includes a Retry mechanism for failed downloads 
 ```
 max_attempts = 3
```
It has 2 Arguments
```
--url_file
```
"Path to the file containing direct download URLs."
```
--max_threads
```
 "Maximum number of concurrent download threads." Default is 5

