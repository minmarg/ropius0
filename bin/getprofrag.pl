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

Get a fragment from COMER profile.
(C)2019 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input COMER profile.

-o <filename>   Name of output COMER profile.

-b <position>   Start position of a profile fragment.
            Default=1

-e <position>   End position of a profile fragment.
            Default=[Profile length]

--norenum       Do not re-enumerate the fragment.

-h              This text.

EOIN

my  $INPUT;
my  $OUTPUT;
my  $BEG = 1;
my  $END = -1;
my  $RENUM = 1;
my  $Fail;

my  $result = GetOptions(
               'i=s'      => \$INPUT,
               'o=s'      => \$OUTPUT,
               'b=i'      => \$BEG,
               'e=i'      => \$END,
               'norenum'  => sub { $RENUM = 0; },
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Missing arguments.\n$usage"; $Fail = 1; } unless( $Fail || ( $INPUT && $OUTPUT ));
do { print STDERR "ERROR: File $INPUT does not exist.\n"; $Fail = 1; } unless( $Fail || -f $INPUT );

do { print STDERR "ERROR: Invalid profile start position given.\n"; $Fail = 1; } unless( $Fail || 0 < $BEG );
do { print STDERR "ERROR: Invalid profile end position given.\n"; $Fail = 1; } unless( $Fail || $END < 0 || $BEG < $END);

exit(1) if $Fail;

my  $length = 0;
my  $profrag = '';

unless( GetProFrag($INPUT, $BEG, $END, $RENUM, \$length, \$profrag)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
unless( WriteProFrag($OUTPUT, \$profrag)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}

printf( STDERR "Profile length, %d\nDone.\n", $length );
exit( 0 );

## ===================================================================
## read and save a profile fragment
##
sub GetProFrag
{
    my  $input = shift;
    my  $beg = shift;
    my  $end = shift;
    my  $renum = shift;
    my  $rlen = shift;##ref
    my  $rfrag = shift;##ref
    my ($ignore, $fail) = (0,0);

    unless( open(IN, $input)) {
        printf( STDERR "ERROR: GetProFrag: Failed to open file $input: $!\n" );
        return 0;
    }

    while(<IN>) {
        $ignore = 0 if /^K,\s+Lambda:/;
        if(/^(\d+)\s+[A-Z]\s+\d+/) {
            $ignore = 0;
            $ignore = 1 if $1 < $beg;
            $ignore = 1 if 0 < $end && $end < $1;
            next if $ignore;
            $$rlen++;
            s/^(\d+)(.+)$/$$rlen$2/ if $renum;
        }
        next if $ignore;
        $$rfrag .= $_;
    }

    close(IN);
    $$rfrag =~ s/(LEN:\s+)(\d+)/$1$$rlen/;
    return 0 if $fail;
    return 1;
}

## -------------------------------------------------------------------
## write the profile fragment to file
##
sub WriteProFrag
{
    my  $output = shift;
    my  $rfrag = shift;##ref
    my  $nents = 0;

    unless( open(F, ">$output")) {
        printf( STDERR "ERROR: Failed to open file for writing: $output: $!\n" );
        return 0;
    }

    print( F $$rfrag);

    close(F);
    return 1;
}

## <<>>
