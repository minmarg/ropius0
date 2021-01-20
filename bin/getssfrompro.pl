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

Extract secondary structure information from COMER profile and write in 
PSIPRED vertical fromat.
(C)2019 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input COMER profile.

-o <filename>   Name of output file of secondary structure information.

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
do { print STDERR "ERROR: Missing arguments.\n$usage"; $Fail = 1; } unless( $Fail || ( $INPUT && $OUTPUT ));
do { print STDERR "ERROR: File $INPUT does not exist.\n"; $Fail = 1; } unless( $Fail || -f $INPUT );

exit(1) if $Fail;

my  $length = 0;
my  @secstr;

unless( GetSS($INPUT, \$length, \@secstr)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
unless( WriteSS($OUTPUT, \@secstr)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}

printf( STDERR "\nSS length, %d\nDone.\n", $length );
exit( 0 );

## ===================================================================
## get secondary structure encapsulated in the profile
##
sub GetSS
{
    my  $input = shift;
    my  $rlen = shift;##ref
    my  $rsstr = shift;##ref
    my ($last, $num, $res ) = ('',0,'X');
    my  $ndx = 1;

    unless( open(IN, $input)) {
        printf( STDERR "ERROR: Failed to open file $input: $!\n" );
        return 0;
    }

    while(<IN>) {
        chomp;
        $last = $_;
        if( $last =~ /^(\d+)\s+([A-Z])\s+\d+/) {
            $num = $ndx++;##$1;
            $res = $2;
        }
        if( $last =~ /^\s+SS:([CEH])\s+(\d+)\s+(\d+)\s+(\d+)/) {
            push @$rsstr, [$num, $res, $1, $2, $4, $3];
            $$rlen++;
        }
    }

    close(IN);
    return 1;
}

## -------------------------------------------------------------------
## write extracted secondary structure to file
##
sub WriteSS
{
    my  $output = shift;
    my  $rsstr = shift;##ref
    my  $scale = 10000;
    my  $rec;

    unless( open(F, ">$output")) {
        printf( STDERR "ERROR: Failed to open file for writing: $output: $!\n" );
        return 0;
    }

    ## required by Rosetta
    print( F "# PSIPRED VFORMAT (PSIPRED V2.6)\n");

    foreach $rec (@$rsstr) {
        printf( F "%4d %1s %1s  %6.3f %6.3f %6.3f\n",
          $$rec[0], $$rec[1], $$rec[2], 
          $$rec[3]/$scale, $$rec[4]/$scale, $$rec[5]/$scale);
    }

    close(F);
    return 1;
}

## <<>>
