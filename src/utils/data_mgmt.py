import logging
import os
import random
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re


def process_posts(fd_in, fd_out_train, fd_out_test, target_tag, split):
    line_num= 1
    for line in tqdm(fd_in):
        try:
            fd_out= fd_out_train if random.random() > split else fd_out_test # if random is grater than split like random is 80 percent then make it train else make it test
            attr= ET.fromstring(line).attrib

            pid= attr.get("Id", "")
            label= 1 if target_tag in attr.get("Tags", "") else 0
            title= re.sub(r"\s+", " ", attr.get("Title", "")).strip()
            body= re.sub(r"\s+", " ", attr.get("Body", "")).strip()

            text= title + " " + body

            fd_out.write(f"{pid}\t{label}\t{text}\n")
            title= re.sub(r"\s+", " ", attr.get("Title", "")).strip()
            line_num += 1
            


        except Exception as e:
            msg= f"Error in line {line_num}: {e}"
            logging.exception(msg)
