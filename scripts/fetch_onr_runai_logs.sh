mkdir -p onr_logs
counter=0
while [ $counter -le 107 ]
do
	runai logs vladi-onr-$counter > onr_logs/vladi-onr-$counter.txt 
	((counter++))
done
