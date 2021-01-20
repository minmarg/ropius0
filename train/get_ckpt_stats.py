#!/usr/bin/env python
## (C)2020 Mindaugas Margelevicius, Vilnius University

import sys, os
import numpy as np
from optparse import OptionParser
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as ickpt

description = "Print statistics of tensor values from the given checkpoint model file."

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--input", dest="input",
                      help="input checkpoint file", metavar="FILE")

    (options, args) = parser.parse_args()

    if not options.input:
        sys.stderr.write("ERROR: Input file not given.\n")
        sys.exit()

    return options


def print_tensor_stats(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False,
                                     count_exclude_pattern=""):
  """[Adapted from the tensorflow package]
  Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
    count_exclude_pattern: Regex string, pattern to exclude tensors when count.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        if all_tensors:
          keyvals = reader.get_tensor(key)
          print('{:<80} {:<20} {:+9.4f} {:+10.4f} {:+9.4f}'.
            format(key,str(np.shape(keyvals)),np.min(keyvals),np.max(keyvals),np.std(keyvals)))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))

  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        any(e in file_name for e in [".index", ".meta", ".data"])):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))



if __name__ == "__main__":
    options = ParseArguments()
    print_tensor_stats(options.input, all_tensors=True, tensor_name='')
#<<>>
