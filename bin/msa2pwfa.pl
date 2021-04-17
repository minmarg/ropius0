#!/usr/bin/perl -w

##
## (C)2020 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use File::Basename;
use Getopt::Long;

my  $MYPROGNAME = basename( $0 );
my  $X = 'X';
my  $EVALUE = 1e-4;
my  $WIDTH = -1;
my  $usage = <<EOIN;

Convert multiple sequence alignment in FASTA to pairwise
sequence alignments in FASTA-like format.
(C)2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input file of multiple sequence alignment.

-o <filename>   Name of output file of pairwise sequence alignments.

-r <reference>  Reference name (ID) of the sequence with respect to which 
                generate pairwise alignments.

-e <e-value>    Read alignments with an expectation value less or equal
                than the specified value.
        default=$EVALUE

-w <width>      Number of characters to wrap aligned sequences at.
        By default, no wrapping is used.

-h              Short description.

EOIN


my  $INPUT;
my  $OUTPUT;
my  $TARGET;
my  $Fail = 0;
my  $processed = 0;

my  $result = GetOptions(
               'i=s'      => \$INPUT,
               'o=s'      => \$OUTPUT,
               'r=s'      => \$TARGET,
               'e=f'      => \$EVALUE,
               'w=i'      => \$WIDTH,
               'help|h'   => sub { print $usage; exit( 0 ); }
);


do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Arguments missing.\n$usage"; $Fail = 1; } unless( $Fail || ($INPUT && $OUTPUT && $TARGET));
do { print STDERR "ERROR: File $INPUT does not exist.\n"; $Fail = 1; } unless( $Fail || -f $INPUT );

## ===================================================================

if( $Fail ) {
    exit( 1 );
}

my ($trgdesc,$trgseqn);

unless(FindTarget($INPUT, $TARGET, \$trgdesc, \$trgseqn)) {
    printf( STDERR "Failed.\n" );
    exit(1);
}
unless($trgdesc && $trgseqn) {
    printf( STDERR "Target not found in MSA.\n" );
    exit(1);
}

unless(AlignTopHits($INPUT, $OUTPUT, $EVALUE, $WIDTH, $TARGET, \$trgdesc, \$trgseqn, \$processed)) {
    printf( STDERR "Failed.\n" );
    exit(1);
}

printf( STDERR "\nTotal sequences, %d\nDone.\n", $processed );
exit(0);

## ===================================================================
## align top hits from blast output
##

sub FindTarget
{
    my  $input = shift;
    my  $target = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $reg;
    ##
    unless( open(F, $input)) {
        printf( STDERR "ERROR: Failed to open $input: $!\n" );
        return 0;
    }
    while(<F>) {
        chomp;
        last if /^>/ && $reg;
        if( /^>($target\s.*)$/ ) {
            $$rtrgdesc = $1;
            $reg = 1;
            next;
        }
        next unless $reg;
        s/\s//g;
        $$rtrgseqn .= uc("$_");
    }
    close(F);
    return 1;
}

## ===================================================================
## align top hits from blast output
##

sub AlignTopHits
{
    my  $input = shift;
    my  $output = shift;
    my  $evalue = shift;
    my  $width = shift;
    my  $target = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $rproc = shift;
    my  %hithash;
    my  %queryin;
    my  $alnmask;
    my  $query = 'theonly';
    my ($rec, $len ) = ( 1, 0 ); ## the first record is reserved

    return 0 unless ExtractAlignments($input, $evalue, $query, \%hithash, $rec);
    return 0 unless PrintAligned($target, $rtrgdesc, $rtrgseqn, \%hithash, $output, $width, $rproc );
    return 1;
}

## -------------------------------------------------------------------
## extract top alignments from input
##

sub ExtractAlignments
{
    my  $input = shift;
    my  $evalue = shift;
    my  $query = shift;
    my  $refhash = shift; ## reference to hash of hits
    my  $stindex = shift; ## start index

    my  $hitnum = 0;
    my  $rec = $stindex;
    my  $last;
    my  $e_val;
    my  $score;
    my  $qname;

    my ($sbjct, $sbjctname, $titletmp, $skip );

    my  $queryfasta;
    my  $sbjctfasta;

    my  $querystart;
    my  $sbjctstart;
    my  $queryend;
    my  $sbjctend;
    my  $sbjctlen;


    unless( open( IN, $input )) {
        printf( STDERR "ERROR: Failed to open $input: $!\n" );
        return 0;
    }
    while( <IN> ) {
##        next if /^$/; ## comment to make eof( IN ) to work
        chomp;
        $last = $_;

        if( $last =~ /^([a-zA-Z\-\.]+)$/) {
            $sbjctfasta .= $1; ##uc("$1");
        }

        if( $last =~ /^>/ || 
            eof( IN ))
        {
            $titletmp = $1;
            $hitnum++;
            $skip = 0;

            if( defined($sbjct) && defined($e_val)) {
                if( !$sbjct ) {
                    printf( STDERR "WARNING: No subject description (Hit no. %d). Skipped.\n", $hitnum-1 );
                    $skip = 1;
                }
                elsif(!defined($e_val)) {
                    printf( STDERR "WARNING: No expect value (Hit no. %d: %s...). Skipped.\n",
                            $hitnum-1, substr($sbjct, 0, 20));
                    $skip = 1;
                }
                unless( $sbjctfasta && length($sbjctfasta)) {
                    printf( STDERR "WARNING: Invalid alignment (sequence lengths, Hit no. %d: %s...). Skipped.\n",
                            $hitnum-1, substr($sbjct, 0, 20));
                    $skip = 1;
                }
                unless( $sbjctstart ) {
                    printf( STDERR "WARNING: No alignment start positions (Hit no. %d: %s...). Skipped.\n",
                            $hitnum-1, substr($sbjct, 0, 20));
                    $skip = 1;
                }
                unless( $sbjctend ) {
                    printf( STDERR "WARNING: No alignment end positions (Hit no. %d: %s...). Skipped.\n",
                            $hitnum-1, substr($sbjct, 0, 20));
                    $skip = 1;
                }

                $sbjctname = ($sbjct =~ /^([\w\.]+)/)? $1: '';
                $skip = 1 if $evalue < $e_val || $e_val < 0.0;

                unless( $skip ) {
                    $$refhash{$query}[$rec][0] = $sbjct;
                    $$refhash{$query}[$rec][1] = $e_val;

                    $$refhash{$query}[$rec][2] = ''; ## reserved
                    $$refhash{$query}[$rec][3] = ''; ## reserved
                    $$refhash{$query}[$rec][4] = '';#$queryfasta;
                    $$refhash{$query}[$rec][5] = $sbjctfasta;

                    $$refhash{$query}[$rec][6] = 0;#$querystart;
                    $$refhash{$query}[$rec][7] = 0;#$queryend;
                    $$refhash{$query}[$rec][8] = $sbjctstart;
                    $$refhash{$query}[$rec][9] = $sbjctend;
                    $$refhash{$query}[$rec][10]= $sbjctlen;

                    $rec++;
                }
            }

            undef $sbjct;
            undef $sbjctlen;
            undef $queryfasta;
            undef $sbjctfasta;
            undef $querystart;
            undef $sbjctstart;
            undef $queryend;
            undef $sbjctend;
            undef $e_val;
            undef $qname;
        }

        if( $last =~ /^>(\S+)(?:.*\s+)(?:\()?(?:ALN:\s+)?(\d+)[\-\s](\d+)(?:\))?(.*)$/) {
            $sbjct = "$1 $4";
            $sbjctstart = $2;
            $sbjctend = $3;
            if($last =~ /[=\s]([\d\.eE\-\+]+|n\/a)$/) {
                $e_val = $1;
                $e_val = "1$e_val" if $e_val =~ /^e/i;
            }
            $sbjctlen = $1 if $last =~ /LEN=(\d+)\s*/;
            undef $sbjctfasta;
            next;
        }
    }

    close( IN );
    return 1;
}

## -------------------------------------------------------------------
## print multiply aligned top hits
##

sub PrintAligned
{
    my  $target = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $rhithash = shift;##reference
    my  $filename = shift;
    my  $width = shift;
    my  $rproc = shift;

    my  $rec;
    my  $e_val;
    my  $query;
    my  $sbjct;
    my  $qufasta;
    my  $sbfasta;

    my  $querystart;
    my  $sbjctstart;
    my  $queryend;
    my  $sbjctend;

    unless( open( OUT, ">$filename" )) {
        printf( STDERR "ERROR: Failed to open $filename for writing.\n" );
        return 0;
    }

    foreach $query( keys %{$rhithash} ) {
        for( $rec = 1; $rec <= $#{$$rhithash{$query}}; $rec++ ) {
            $sbjct      = $$rhithash{$query}[$rec][0];
            $e_val      = $$rhithash{$query}[$rec][1];
            ##$qufasta   = \$$rhithash{$query}[$rec][4];
            $sbfasta   = \$$rhithash{$query}[$rec][5];

            ##$querystart = $$rhithash{$query}[$rec][6];
            ##$queryend   = $$rhithash{$query}[$rec][7];
            $sbjctstart = $$rhithash{$query}[$rec][8];
            $sbjctend   = $$rhithash{$query}[$rec][9];

            next if $sbjct =~ /^$target/;

            return 0 unless 
                PrintAlignedHelperPass2(\*OUT, $rtrgdesc, $rtrgseqn, $width, $sbjct, $sbjctstart, $sbjctend, $sbfasta, $e_val, $rec);
            $$rproc++;
        }
    }

    close( OUT );
    return 1;
}

sub PrintAlignedHelperPass2
{
    my  $reffile = shift;##file reference
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $width = shift;
    my  $sbjct = shift;
    my  $sbjctstart = shift;
    my  $sbjctend = shift;
    my  $sbfasta = shift;
    my  $e_val = shift;
    my  $rec = shift;

    my  $trgfastacopy = $$rtrgseqn;
    my  $sbtfastacopy = $$sbfasta;

    if(!length($sbtfastacopy) || length($trgfastacopy) != length($sbtfastacopy)) {
        printf(STDERR "ERROR: Inconsistent target and template (%s) lengths: %d vs %d\n",
            $sbjct, length($trgfastacopy), length($sbtfastacopy));
        return 0;
    }

    for(my $i = length($trgfastacopy)-1; 0 <= $i; $i--) {
        if(substr($trgfastacopy,$i,1) eq '-' && substr($sbtfastacopy,$i,1) eq '-') {
            substr($trgfastacopy,$i,1,'');
            substr($sbtfastacopy,$i,1,'');
        }
    }

    printf($reffile ">%s\n", $$rtrgdesc );
    WrapFasta( $reffile, \$trgfastacopy, $width );

    ##$sbjct =~ s/^(\S+)\s*(.*)$/$1 ($sbjctstart-$sbjctend) $2  Expect=$e_val/ if $rec;
    $sbjct =~ s/^(\S+)\s*(.*)$/$1 ($sbjctstart-$sbjctend) $2/;
    printf($reffile ">%s\n", $sbjct );
    WrapFasta( $reffile, \$sbtfastacopy, $width );

    print($reffile "//\n");

    return 1;
}

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## -------------------------------------------------------------------
## wrap sequence to fragments of equal length
##

sub WrapFasta {
    my  $reffile = shift;   ## reference to file descriptor
    my  $reffasta = shift;  ## reference to sequence
    my  $width = shift;     ## width of fragment per line
    my  $padding = 0;       ## padding at the beginning of each line
    my  $line;

    $width = 999999 if $width <= 0;

    if( ref( $reffile ) ne 'GLOB' && ref( $reffile ) ne 'SCALAR' ) {
        printf( STDERR "ERROR: WrapFasta: Wrong reference.\n" );
        return 0;
    }

##    $$reffile = '' if( ref( $reffile ) eq 'SCALAR' );

    for( my $n = 0; $n < length( $$reffasta ); $n += $width ) {
        if( $n && $padding ) {
            $line = sprintf( "%${padding}s%s\n", ' ', substr( $$reffasta, $n, $width ));
        } else {
            $line = sprintf( "%s\n", substr( $$reffasta, $n, $width ));
        }
        if( ref( $reffile ) eq 'SCALAR' ) {
                 $$reffile .= $line;
        } else { printf( $reffile $line );
        }
    }
    return 1;
}

## <<>>

