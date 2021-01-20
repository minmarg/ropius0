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

Make BLAST checkpoint file in its legacy binary format, using relevant 
information extracted from COMER profile.
(C)2019 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input COMER profile.

-o <filename>   Name of output BLAST checkpoint file.

-p <filename>   Name of output BLAST pssm file.
        optional

-h              This text.

EOIN

my  $INPUT;
my  $OUTPUT;
my  $PSSM;
my  $Fail;

my  $result = GetOptions(
               'i=s'      => \$INPUT,
               'o=s'      => \$OUTPUT,
               'p=s'      => \$PSSM,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Missing arguments.\n$usage"; $Fail = 1; } unless( $Fail || ( $INPUT && $OUTPUT ));
do { print STDERR "ERROR: File $INPUT does not exist.\n"; $Fail = 1; } unless( $Fail || -f $INPUT );

exit(1) if $Fail;

my  $LMB = 0.2305;##Lambda
## Robinson & Robinson background probabilities (PNAS USA 88, 1991, 8880-4)
my  @BP = (0.07805,0.05129,0.04487,0.05364,0.01925,0.04264,0.06295,0.07377,0.02199,
           0.05142,0.09019,0.05744,0.02243,0.03856,0.05203,0.07120,0.05841,0.01330,
           0.03216,0.06441);

my  $length = 0;
my  %chkstr;

unless( GetChk($INPUT, $LMB, \@BP, \$length, \%chkstr)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
unless( WriteChk($OUTPUT, \%chkstr)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}
if( $PSSM && !WritePssm($PSSM, \%chkstr)) {
    printf( STDERR "Failed.\n" );
    exit( 1 );
}

printf( STDERR "Profile length, %d\nDone.\n", $length );
exit( 0 );

## ===================================================================
## get relevant information and produce scores for chk file
##
sub GetChk
{
    my  $input = shift;
    my  $lambda = shift;
    my  $rbps = shift;##ref
    my  $rlen = shift;##ref
    my  $rstr = shift;##ref
    my  $scale = 10000.;
    my  $iscale = 1/$scale;
    my ($last, $seq, $prb, $fail );

    unless( open(IN, $input)) {
        printf( STDERR "ERROR: GetChk: Failed to open file $input: $!\n" );
        return 0;
    }

    while(<IN>) {
        chomp;
        $last = $_;
        if( $last =~ /^\d+\s+([A-Z])(\s+\d+.+)$/) {
            my @prbs;
            $seq .= $1;
            $prb = $2;
            push @prbs, $1*$iscale while( $prb=~/\s+(\d+)/g);
            if( $#prbs != 19 ) {
                printf( STDERR "ERROR: GetChk: Invalid number of probabilities: pos %d\n", $$rlen );
                $fail = 1;
                last;
            }
            foreach(@prbs) {
                if( 1. <= $_ || $_ <= 0.) {
                    printf( STDERR "ERROR: GetChk: Invalid probability: %d, pos %d\n", $_, $$rlen );
                    $fail = 1;
                    last;
                }
            }
            push @{$$rstr{DATA}}, log($prbs[$_]/$$rbps[$_])/$lambda foreach(0..19);
            push @{$$rstr{TRGF}}, int($prbs[$_]*100.) foreach(0..19);
            $$rlen++;
        }
    }

    close(IN);
    return 0 if $fail;
    $$rstr{LEN} = $$rlen;
    $$rstr{SEQ} = $seq;
    return 1;
}

## -------------------------------------------------------------------
## write obtained scores to binary chk file
##
sub WriteChk
{
    my  $output = shift;
    my  $rstr = shift;##ref
    my  $nents = 0;

    unless( open(F, ">$output")) {
        printf( STDERR "ERROR: Failed to open file for writing: $output: $!\n" );
        return 0;
    }
    binmode(F) or do {
        printf( STDERR "ERROR: Failed to set binary mode for output file: $!\n" );
        return 0;
    };

    $nents = scalar(@{$$rstr{DATA}});

    print( F pack("i", $$rstr{LEN}));
    print( F $$rstr{SEQ});
    print( F pack("d$nents", @{$$rstr{DATA}}));

    close(F);
    return 1;
}

## -------------------------------------------------------------------
## write score matrix to pssm file
##
sub WritePssm
{
    my  $output = shift;
    my  $rstr = shift;##ref
    my  $len = $$rstr{LEN};
    my  $effnoress = 20;
    my  $n = 0;

    unless( open(F, ">$output")) {
        printf( STDERR "ERROR: Failed to open file for writing: $output: $!\n" );
        return 0;
    }

    printf( F "Position-specific scoring matrix computed, weighted observed percentages rounded down, ".
              "information per position, and relative weight of gapless real matches to pseudocounts\n".
              "%12sA   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V".
              "   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n"," ");

    for( my $i = 0; $i < $len; $i++ ) {
        printf( F "%5d %1s  ", $i+1, substr($$rstr{SEQ}, $i, 1));
        for( my $j = 0; $j < $effnoress; $j++ ) {
            printf( F " %3d", $$rstr{DATA}[$n+$j]);
        }
        printf( F " ");
        for( my $j = 0; $j < $effnoress; $j++ ) {
            printf( F " %3d", $$rstr{TRGF}[$n+$j]);
        }
        printf( F " %5.2f %8.2f\n", 0., 0.);
        $n += $effnoress;
    }

    printf( F "\n                      K         Lambda\n");

    close(F);
    return 1;
}

## <<>>
