#!/usr/bin/perl -w

##
## (C)2019 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use File::Basename;
use Getopt::Long;

my  $MYPROGNAME = basename( $0 );
my  $usage = <<EOIN;

Extract amino acid sequence from COMER profile.
(C)2019 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input COMER profile.

-o <filename>   Name of output file.

-h              This text.

EOIN

my  $INPUT;
my  $OUTPUT;
my  $Fail;

my  $result = GetOptions(
               'i=s'      => \$INPUT,
               'o=s'      => \$OUTPUT,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
#do { print STDERR "ERROR: Missing arguments.\n$usage"; $Fail = 1; } unless( $Fail || ( $INPUT ));
#do { print STDERR "ERROR: File $INPUT does not exist.\n"; $Fail = 1; } unless( $Fail || -f $INPUT );

exit(1) if $Fail;

my  $length = 0;
my ($name, $seqn);

unless( GetSeq($INPUT, \$name, \$seqn)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
unless( WriteSeq($OUTPUT, $name, \$seqn)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}

exit( 0 );

## ===================================================================
## get sequence of the profile
##
sub GetSeq
{
    my  $input = shift;
    my  $rname = shift;##ref to name
    my  $rsstr = shift;##ref
    my ($fh, $inputname);
    ##
    $inputname = $input? $input: 'STDIN';
    $fh = \*STDIN unless $input;
    if($input && !open($fh, $input)) {
        print(STDERR "ERROR: Failed to open file: $input: $!\n");
        return 0;
    }

    $$rsstr = '';

    while(<$fh>) {
        chomp;
        $$rname = $1 if /^DESC:\s+(\S+)/;
        $$rsstr .= $2 if /^(\d+)\s+([A-Z])\s+\d+/;
    }

    close($fh) if $input;
    return 1;
}

## -------------------------------------------------------------------
## write sequence to file
##
sub WriteSeq
{
    my  $output = shift;
    my  $name = shift;
    my  $rsstr = shift;##ref
    my  $F = \*STDOUT;
    my  $rec;

    if( $output && !open($$F, ">$output")) {
        printf( STDERR "ERROR: Failed to open file for writing: $output: $!\n" );
        return 0;
    }

    printf($F ">%s (%d)\n%s\n", $name, length($$rsstr), $$rsstr);

    close($F) if $output;
    return 1;
}

## <<>>
