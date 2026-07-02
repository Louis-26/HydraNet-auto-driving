# Linux Environment 
If you are in linux and want to use the script, 
please add:
```bash
sed -i 's/\r$//' git_script/*.sh
```


# lfs test

Currently, git lfs remove remotely has been verified✅


# test for lfs push and remove
## test 1-remove single file
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_2.txt 107374182 # 102.4 MB
```
## test 2-remove folder
```bash
mkdir file_1 
fsutil file createnew file_1/test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_2.md 107374182 # 102.4 MB
```

```bash
mkdir file_2
fsutil file createnew file_2/test_lf_3.sh 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_4.mp4 107374182 # 102.4 MB
```
## test 3-remove with ext
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_2.txt 107374182 # 102.4 MB
mkdir file_1 
fsutil file createnew file_1/test_lf_3.txt 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_4.txt 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_5.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_6.txt 107374182 # 102.4 MB
mkdir file_2
fsutil file createnew file_2/test_lf_7.txt 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_8.txt 107374182 # 102.4 MB
```

## test 4-remove all
```bash
fsutil file createnew test_lf_1.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_2.md 107374182 # 102.4 MB
mkdir file_1
fsutil file createnew file_1/test_lf_3.jpg 107374182 # 102.4 MB
fsutil file createnew file_1/test_lf_4.mp4 107374182 # 102.4 MB
```

```bash
fsutil file createnew test_lf_5.txt 107374182 # 102.4 MB
fsutil file createnew test_lf_6.md 107374182 # 102.4 MB
mkdir file_2
fsutil file createnew file_2/test_lf_7.jpg 107374182 # 102.4 MB
fsutil file createnew file_2/test_lf_8.mp4 107374182 # 102.4 MB
```
