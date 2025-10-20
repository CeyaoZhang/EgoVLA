#/bin/bash

for t in tests/bash/test_*.sh; do
    echo "========================== Testing $t =================================="
    bash $t;
    if [ $? != 0 ];
    then
        echo "$t fails"
        exit -1
    fi
done
