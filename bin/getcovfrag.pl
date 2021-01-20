#!/usr/bin/perl -w

##
## (C)2020 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use File::Basename;
use Getopt::Long;

my  $MYPROGNAME = basename( $0 );
my  $usage = <<EOIN;

Extract a fragment from COMER xcov file.
(C)2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input COMER xcov file.

-o <filename>   Name of output COMER xcov file.

-b <position>   Start position of a fragment of a COMER profile 
                (for which xcov pairwise information has been obtained).
            Default=1

-e <position>   End position of a COMER profile fragment.
            Default=[Profile length]

--norenum       Do not re-enumerate the output.

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
my  $filefrag = '';

unless( GetFileFrag($INPUT, $BEG, $END, $RENUM, \$length, \$filefrag)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
unless( WriteFileFrag($OUTPUT, \$filefrag)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}

printf( STDERR "Profile length, %d\nDone.\n", $length );
exit( 0 );

## ===================================================================
## read and save an xcov file fragment
##
sub GetFileFrag
{
    my  $input = shift;
    my  $beg = shift;
    my  $end = shift;
    my  $renum = shift;
    my  $rlen = shift;##ref
    my  $rfrag = shift;##ref
    my ($pos1, $pos2) = (0,0);
    my ($ignore, $fail) = (0,0);

    unless( open(IN, $input)) {
        printf( STDERR "ERROR: GetFileFrag: Failed to open file $input: $!\n" );
        return 0;
    }

    while(<IN>) {
        if(/^(\d+)\s+(\d+)\s+/) {
            $ignore = 0;
            $ignore = 1 if $1 < $beg || $2 < $beg;
            $ignore = 1 if 0 < $end && ($end < $1 || $end < $2);
            next if $ignore;
            $pos1 = $1 - $beg + 1;
            $pos2 = $2 - $beg + 1;
            $$rlen = $pos1;
            s/^(\d+)(\s+)(\d+)(.+)$/$pos1$2$pos2$4/ if $renum;
        }
        next if $ignore;
        $$rfrag .= $_;
    }

    close(IN);
    $$rfrag =~ s/(Length=\s+)(\d+)/$1$$rlen/;
    return 0 if $fail;
    return 1;
}

## -------------------------------------------------------------------
## write the xcov file fragment to file
##
sub WriteFileFrag
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
