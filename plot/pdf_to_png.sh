for i in *.pdf
do
  filename=${i%.*}
  sips -s format png ${filename}.pdf --out ${filename}.png
done
