##utils.py

import hashlib
import os


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files. This is, files with
    extension ".png", ".jpg", ".jpeg" or ".gif".

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    yes_extension = ['.png','.jpg','.jpeg','.gif']
    name, extension = os.path.splitext(filename)
    if extension.lower() in yes_extension:
        return True
    


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    #with open(file, 'rb') as f:
      #  contents = f.read()
       # filename_content = hashlib.md5(contents)
       # file.filename = filename_content.hexdigest() + '.' + file.filename.split('.')[1]

    extension = os.path.splitext(file.filename)[1]
    # Create the hashed name
    hashed_name = hashlib.md5(file.stream.read())
    # Recover readability of file by pointing to first character
    file.stream.seek(0)

    return hashed_name.hexdigest() + extension
  


    #return os.path.basename(file.filename)
