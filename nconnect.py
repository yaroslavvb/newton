#!/usr/bin/env python
# connects to instance

import argparse
import fcntl
import logging
import os
import select
import subprocess
import sys

from ncluster import aws_util as u

parser = argparse.ArgumentParser(description='sync')
parser.add_argument("name", help="name of machine")
parser.add_argument('-v', '--verbose', action='count', dest='verbosity',
                    default=0, help='Set verbosity.')
args = parser.parse_args()


def main():
  instances = u.lookup_instances(args.name, verbose=False)
  if not instances:
    assert False, f"Couldn't find any instances containing {args.name}"
  instance = instances[0]
  print(f"Connecting to {u.get_name(instances[0])}")
  cmd = f"ssh -i {u.get_keypair_fn()} -o StrictHostKeyChecking=no ubuntu@{instance.public_ip_address}"
  print(cmd)
  os.system(cmd)

if __name__ == '__main__':
  sys.exit(main())
