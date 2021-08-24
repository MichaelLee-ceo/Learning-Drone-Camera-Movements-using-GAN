@echo off
echo ^> Downloading models...
curl http://140.138.152.104:6969/models/models.zip --output models.zip
echo ^> Zipping model files

tar -zxvf models.zip -C "./"

del models.zip

echo ^> ***All files zipped***
pause