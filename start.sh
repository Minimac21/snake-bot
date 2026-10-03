if [[ -e "out/frame0.jpg" || -e "out/bg0.jpg" ]]; then
	rm out/*.jpg
fi

echo "per tick, objects sorted by x increasing" > out/objects.txt

python bot.py $@