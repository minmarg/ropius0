#!/usr/bin/env perl

use strict;
use File::Basename;
use Getopt::Long;

my  $MYPROGNAME = basename($0);

my  $usage = <<EOIN;

Extract a profile from a COMER profile database.
(C)2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   COMER profile database with extension.

-n <name>       Name of the profile (FILE: <name>) to extract it from the database.

EOIN

my  $COMERDB;
my  $NAME;
my  $Fail = 0;

my  $result = GetOptions(
               'i=s'      => \$COMERDB,
               'n=s'      => \$NAME,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Arguments missing.\n$usage"; $Fail = 1; } unless( $Fail || ( $COMERDB && $NAME ));
do { print STDERR "ERROR: File $COMERDB does not exist.\n"; $Fail = 1; } unless( $Fail || -f $COMERDB );

exit(1) if($Fail);

my ($t, $d);

unless( open( F, $COMERDB )) {
    printf( STDERR "ERROR: Failed to open $COMERDB: $!\n" );
    exit(1);
}

while(<F>){
    if(/^COMER\s+profile/){
        $t=$_;
        next
    }
    $d=1 if /^FILE:\s+$NAME/;
    if($d || /^DESC:/){
        $t.=$_;
    }
    last if($d && /^\*/);
}

close(F);

print $t

