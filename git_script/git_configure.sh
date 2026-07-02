# in this file, common git configuration is set
# user name and email
git config --global user.name "Yi Lu"
git config --global user.email "121090386@link.cuhk.edu.cn"

# ensure LF is always the end of line character, not matter windows or linux
git config --global core.autocrlf input 

# set the buffer size to 2GB
git config --global http.postBuffer 524288000

# Default branch name
git config --global init.defaultBranch main