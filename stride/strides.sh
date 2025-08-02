wget https://webclu.bio.wzw.tum.de/stride/stride.tar.gz --directory-prefix src

cd src || exit

tar -zxf stride.tar.gz

make

cd ..

cp src/stride .

rm -rf src
