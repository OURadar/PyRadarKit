#!/bin/bash

file=$(find log -name 'pyrk*.log' | sort | tail -n 1)

clear

if [ -f ${HOME}/bin/blib.sh ]; then
	. ${HOME}/bin/blib.sh
	check_process "python main.py" "main.py"
	echo ""
	echo "" | textout "Log = ${file}" green
else
	echo ""
	echo "${file}" 
fi

tput sgr0
if [ ! -z ${file} ]; then
	tail -n 40 ${file}
else
	echo "Unable to find a log file."
fi
