#!/bin/bash

CODEC=
TILE=
CRF=
OUTPUT=
PRIMARY=false
HIDDEN=false
THUMB=false
NAME=
MIME=
ITEM=
ITEM_PATH=

for i in "$@"
do
  case ${i} in
      --codec=*)
      CODEC="${i#*=}"
      echo "CODEC = [${CODEC}]"
      ;;
      --tile=*)
      TILE="${i#*=}"
      echo "TILE = [${TILE}]"
      ;;
      --crf=*)
      CRF="${i#*=}"
      echo "CRF = [${CRF}]"
      ;;

      --output=*)
      OUTPUT="${i#*=}"
      ;;

      --primary)
      PRIMARY=true
      ;;
      --hidden)
      HIDDEN=true
      ;;
      --thumb)
      THUMB=true
      ;;
      --name=*)
      NAME="${i#*=}"
      ;;

      --item=*)
      ITEM="${i#*=}"
      ITEM_PATH="${ITEM#*path=}"

      echo .

      echo "PRIMARY = [${PRIMARY}]"
      echo "HIDDEN = [${HIDDEN}]"
      echo "THUMB = [${THUMB}]"
      echo "NAME = [${NAME}]"
      echo "MIME = [${MIME}]"
      echo "ITEM = [${ITEM}]"
      echo "ITEM_PATH = [${ITEM_PATH}]"
      echo "OUTPUT = [${OUTPUT}]"

      if [[ ${PRIMARY} = true ]]; then
        echo "cp -a ${ITEM_PATH} ${OUTPUT}"
        cp -a ${ITEM_PATH} ${OUTPUT}
      else
        echo "cat ${ITEM_PATH} >> ${OUTPUT}"
        cat ${ITEM_PATH} >> ${OUTPUT}
      fi

      PRIMARY=false
      HIDDEN=false
      THUMB=false
      NAME=
      MIME=
      ITEM=
      ITEM_PATH=
      ;;
      --mime=*)
      MIME="${i#*=}"
      ;;

      *)
      echo Unknown option ${i}
      exit 1
      ;;
esac
done
