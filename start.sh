if [[ -e "out/frame0.jpg" || -e "out/bg0.jpg" ]]; then
	rm out/*.jpg
fi

echo "" > out/blobs.txt

python bot.py $@