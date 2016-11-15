#!/bin/bash

cat fail.txt|while read line

do
	echo $line
	./bin/demo $line
done
