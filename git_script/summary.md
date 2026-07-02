It collects all command git command scripts.

large files test link:

https://1024terabox.com/s/1V6pxnMfn_k99WHca05GTtw 

Extraction Code:wkft

# basic command

## push local changes to remote

`git_push.sh`


## pull between branch, from one remote branch to another local branch(replace local files by remote files with the same name)

`git_pull_between_branch.sh`

## pull request, from one remote branch to another remote branch(pull request record on github)

`git_pull_request.sh`

## fetch remote files and download locally
`git_fetch_pull.sh`

## stop tracking a file or folder with git
`git_stop_tracking.sh`

## update all scripts
`git_update_scripts.sh`

# lfs
## push large files or folders to remote repository
`git_lfs_push.sh`

## remove large files or folders from remote repository
`git_lfs_remove.sh`

## restore from a lfs repo to a normal git repo
We usually run this script after running `git_lfs_remove.sh` to completely remove lfs tracking from a repository.
`git_migrate.sh`

## verify that lfs files have been removed from remote repository
`git_lfs_remove_verify_final.sh`
`git_lfs_remove_verify_lf_orphaned.sh`

# complementary operations
## copy repository from one owner to another owner(or another repo name)
`git_copy.sh`

## configure git
`git_config.sh`

## ignore all large files in a repository
`git_ignore_all_large.sh`

## consolidate .gitignore file by replacing individual entries with folder entries, and sort the .gitignore file alphabetically
`gitignore_consolidate.sh`

## remove files in gitignore
`gitignore_remove.sh`

## search all git repos under a path
`gitrepo_search.sh`

## visualize all files with size in a tree structure 
`git_file_tree.sh`

# linux machine change permission

in linux, add `chmod -R +x git_script` for a folder, or `chmod +x git_script/git_push.sh` for a single file, to make it
executable.

# other command

in linux, convert dos to unix: `sed -i 's/\r$//' FILE_NAME`, e.g., `sed -i 's/\r$//' git_script/git_push.sh`

if you meet fatal error on divergent branches when pulling,
use `git config pull.rebase false`
