cat out/objects.txt
sxiv $(for x in out/*.jpg; do printf '%s\n' "$x"; done | sort -V)