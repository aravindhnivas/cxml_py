if [ $# -eq 0 ]
then
  echo "No tags specified"
  exit 1
fi

for tag in "$@"
do
  git tag -d $tag
  git push --delete origin $tag
done
#  % ./git-delete-tags.sh v0.0.2-test2  v0.0.2-test3 v0.0.2-test4 v0.0.2-test5 v0.0.2-test6 