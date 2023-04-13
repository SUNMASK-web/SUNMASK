file=$1
tempo=${2:-160}
timidity -T $tempo --output-24bit -Ow $file
bn=${file%.mid}
ffmpeg -y -i $bn.wav -acodec pcm_s16le -ar 44100 $bn_1.wav
mv $bn_1.wav $bn.wav
ffmpeg -y -i $bn.wav $bn.mp3
