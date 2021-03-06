#!/usr/bin/env python
# Sync utility, forked from original by gdb@openai
# Syncs current directory with /ncluster/newton

import argparse
import fcntl
import logging
import os
import select
import subprocess
import sys
import time

from ncluster import aws_util as u

parser = argparse.ArgumentParser(description='sync')
parser.add_argument('-v', '--verbose', action='count', dest='verbosity',
                    default=0, help='Set verbosity.')
parser.add_argument('-n', '--name', type=str, default='', help="name of instance to sync with")
args = parser.parse_args()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
if args.verbosity == 0:
  logger.setLevel(logging.INFO)
elif args.verbosity >= 1:
  logger.setLevel(logging.DEBUG)


class Error(Exception):
  pass


class Resyncd(object):
  def __init__(self, remote, sync):
    self.remote = remote
    self.sync = sync
    self.counter = 0

  def run(self):
    self.resync()
    sources = [sync.source for sync in self.sync]
    fswatch = subprocess.Popen(['fswatch'] + sources, stdout=subprocess.PIPE)
    fl = fcntl.fcntl(fswatch.stdout.fileno(), fcntl.F_GETFL)
    fcntl.fcntl(fswatch.stdout.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
    while True:
      r, _, _ = select.select([fswatch.stdout], [], [])
      fswatch_output = r[0].read()
      output = fswatch_output.decode('ascii')
      files = output.strip().split("\n")
      # Ignore emacs swap files
      files = [f for f in files if '#' not in os.path.basename(f)]

      # ignore Tensorboard local runs directory
      files = [f for f in files if 'tfevents' not in os.path.basename(f)]
      
      if files:
        print("changed: " + str(files))
      files = set(files)  # remove duplicates from fswatch_output
      if not files:
        continue

      print("---")
      print(files)
      print("---")
      self.resync()

  def resync(self):
    procs = []
    for sync in self.sync:
      instances = u.lookup_instances(args.name, verbose=False)
      if not instances:
        assert False, f"Couldn't find any instances containing {args.name}"
      instance = instances[0]
      print("Syncing with ", u.get_name(instance))

      command = sync.command(instance)
      popen = subprocess.Popen(command)
      procs.append({
        'popen': popen,
        'command': command,
      })
    # Wait
    for proc in procs:
      print(proc["command"])
      proc['popen'].communicate()
    for proc in procs:
      if proc['popen'].returncode != 0:
        raise Error('Bad returncode from %s: %d', proc['command'], proc['popen'].returncode)
    logger.info('Resync %d complete', self.counter)
    self.counter += 1


class Sync(object):
  # todo: exclude .#sync.py
  excludes = ['*.model', '*.cache', '.picklecache', '.git', '*.pyc', '*.gz']

  def __init__(self, source, dest, modify_window=True, copy_links=False, excludes=()):
    self.source = os.path.expanduser(source)
    self.dest = dest
    self.modify_window = modify_window
    self.copy_links = copy_links
    self.excludes = self.excludes + list(excludes)

  def command(self, instance):
    excludes = []
    for exclude in self.excludes:
      excludes += ['--exclude', exclude]

    # todo, rename no_strict_checking to ssh_command

    keypair_fn = u.get_keypair_fn()
    username = 'ubuntu'
    ip = instance.public_ip_address

    ssh_command = "ssh -i %s -o StrictHostKeyChecking=no" % (keypair_fn,)
    no_strict_checking = ['-arvce', ssh_command]

    command = ['rsync'] + no_strict_checking + excludes
    if self.modify_window:
      command += ['--update', '--modify-window=600']
    if self.copy_links:
      command += ['-L']
    command += ['-rv', self.source, username + "@" + ip + ':' + self.dest]
    print("Running ")
    print(command)
    return command


def main():
  sync = [Sync(source='.', dest='/ncluster/newton', copy_links=False), ]

  # obtain ssh
  resyncd = Resyncd('asdf', sync)

  while True:
    try:
      resyncd.run()
    except Exception as e:
      print("Exception", e, "Retrying in 60")
      time.sleep(30)
    
  return 0


if __name__ == '__main__':
  sys.exit(main())
