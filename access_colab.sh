#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  xdg-open https://colab.research.google.com/drive/1XQFCfYi577bS-41wss9kIQx0WnHXgZm7#scrollTo=oWkkzfthGghS
  sleep 3600
done
