#!/usr/bin/perl -w

##
## (C)2020 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use File::Basename;
use Getopt::Long;
use List::Util qw(max);

my  $MYPROGNAME = basename( $0 );
my  $X = 'X';
my  $WIDTH = -1;
my  $usage = <<EOIN;

Sort multiple sequence alignment in FASTA by template 
name and its significance.
(C)2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters>

Parameters:

-i <filename>   Input file of multiple sequence alignment.

-o <filename>   Name of output MSA file.

-r <reference>  Reference name (ID) of the sequence to put it in the 
                first place.

-g              Do not delete columns with gaps in the reference.

-w <width>      Number of characters to wrap aligned sequences at.
        By default, no wrapping is used.

-h              Short description.

EOIN


my  $INPUT;
my  $OUTPUT;
my  $TARGET;
my  $KEEPDEL;
my  $Fail = 0;
my  $processed = 0;

my  $result = GetOptions(
               'i=s'      => \$INPUT,
               'o=s'      => \$OUTPUT,
               'r=s'      => \$TARGET,
               'g'        => sub {$KEEPDEL=1;},
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

unless(AlignTopHits($INPUT, $OUTPUT, $WIDTH, $TARGET, $KEEPDEL, \$trgdesc, \$trgseqn, \$processed)) {
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
    my  $width = shift;
    my  $target = shift;
    my  $keepdel = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $rproc = shift;
    my  %hithash;
    my  %queryin;
    my  $alnmask;

    return 0 unless ExtractAlignments($input, \%hithash);
    return 0 unless PrintAligned($target, $keepdel, $rtrgdesc, $rtrgseqn, \%hithash, $output, $width, $rproc);
    return 1;
}

## -------------------------------------------------------------------
## extract top alignments from input
##

sub ExtractAlignments
{
    my  $input = shift;
    my  $refhash = shift; ## reference to hash of hits

    my  $hitnum = 0;
    my  $rec;
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

                unless( $skip ) {
                    $rec = $#{$$refhash{$sbjctname}} + 1;

                    $$refhash{$sbjctname}[$rec][0] = $sbjct;
                    $$refhash{$sbjctname}[$rec][1] = $e_val;

                    $$refhash{$sbjctname}[$rec][2] = ''; ## reserved
                    $$refhash{$sbjctname}[$rec][3] = ''; ## reserved
                    $$refhash{$sbjctname}[$rec][4] = '';#$queryfasta;
                    $$refhash{$sbjctname}[$rec][5] = $sbjctfasta;

                    $$refhash{$sbjctname}[$rec][6] = 0;#$querystart;
                    $$refhash{$sbjctname}[$rec][7] = 0;#$queryend;
                    $$refhash{$sbjctname}[$rec][8] = $sbjctstart;
                    $$refhash{$sbjctname}[$rec][9] = $sbjctend;
                    $$refhash{$sbjctname}[$rec][10]= $sbjctlen;
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
            $sbjct = "$1$4";
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
    my  $keepdel = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $rhithash = shift;##reference
    my  $filename = shift;
    my  $width = shift;
    my  $rproc = shift;

    my ($rec, $len);
    my  $e_val;
    my  $name;
    my  $sbjct;
    my  $qufasta;
    my ($sbfasta, $uninf);
    my ($cnhead,$consensus)=("CONSENSUS",'');
    my  %seqmatches;##sequences' matches to the consensus sequence

    my  $querystart;
    my  $sbjctstart;
    my  $queryend;
    my  $sbjctend;

    unless( open( OUT, ">$filename" )) {
        printf( STDERR "ERROR: Failed to open $filename for writing.\n" );
        return 0;
    }

    my @sbjctnames = 
      sort {max( map{$_->[1]}@{$$rhithash{$b}} ) <=> max( map{$_->[1]}@{$$rhithash{$a}} )} keys %{$rhithash};

    $sbfasta = \$$rhithash{$sbjctnames[0]}[0][5];
    $len = length($$sbfasta);
    $consensus = '-'x$len;

    for(my $i = $len-1; 0 <= $i; $i--) {
        $uninf = 1;
        my %hres;
        my ($nres,$cres) = (0,'X');
        foreach $name( @sbjctnames ) {
            foreach $rec(0..$#{$$rhithash{$name}}) {
                $sbjct = $$rhithash{$name}[$rec][0];
                $sbfasta = \$$rhithash{$name}[$rec][5];
                if($len != length($$sbfasta) || $len != length($$rtrgseqn)) {
                    printf(STDERR "ERROR: (1) Inconsistent target and template (%s) lengths: %d vs %d\n",
                        $sbjct, $len, length($$sbfasta));
                    close(OUT);
                    return 0;
                }
                my $res = substr($$sbfasta,$i,1);
                $uninf = 0 if $res ne '-';
                $nres++;
                $hres{$res}++;
            }
        }
        if(exists($hres{'-'}) && $hres{'-'} > $nres/2) { $cres = '-'; }
        else {
            my @rkeys = sort {$hres{$b} <=> $hres{$a}} keys %hres;
            $cres = ($rkeys[0] eq '-')? $rkeys[1]: $rkeys[0];
            $cres = ($hres{$cres} > $nres/2)? uc($cres): lc($cres);
        }
        substr($consensus,$i,1,$cres);
        if($uninf) {
            ##prune column
            $len--;
            substr($$rtrgseqn,$i,1,'');
            substr($consensus,$i,1,'');
            foreach $name( @sbjctnames ) {
                foreach $rec(0..$#{$$rhithash{$name}}) {
                    $sbfasta = \$$rhithash{$name}[$rec][5];
                    substr($$sbfasta,$i,1,'');
                }
            }
        }
        else {
            ##compute statistics over sequences
            foreach $name( @sbjctnames ) {
                foreach $rec(0..$#{$$rhithash{$name}}) {
                    $sbjct = $$rhithash{$name}[$rec][0];
                    $sbfasta = \$$rhithash{$name}[$rec][5];
                    my $res = substr($$sbfasta,$i,1);
                    next if $cres eq '-';
                    my $sname = $sbjct;
                    $sname =~ s/^(\S+).*$/$1/;
                    $seqmatches{$sname}++ if $cres eq $res;
                }
            }
        }
    }

    my $gapless = $consensus;
    $gapless =~ s/\-//g;
    return 0 unless
        PrintAlignedHelperPass2(\*OUT, $keepdel, $rtrgdesc, $rtrgseqn, $width, $cnhead, 1, length($gapless), \$consensus, 0, $$rproc);
    $$rproc++;

    foreach $name( @sbjctnames ) {
        foreach $rec(sort {$$rhithash{$name}[$a][1] <=> $$rhithash{$name}[$b][1]} 0..$#{$$rhithash{$name}}) {
            $sbjct      = $$rhithash{$name}[$rec][0];
            $e_val      = $$rhithash{$name}[$rec][1];
            ##$qufasta   = \$$rhithash{$name}[$rec][4];
            $sbfasta   = \$$rhithash{$name}[$rec][5];

            ##$querystart = $$rhithash{$name}[$rec][6];
            ##$queryend   = $$rhithash{$name}[$rec][7];
            $sbjctstart = $$rhithash{$name}[$rec][8];
            $sbjctend   = $$rhithash{$name}[$rec][9];

            next if $name =~ /^$target/;

            return 0 unless 
                PrintAlignedHelperPass2(\*OUT, $keepdel, $rtrgdesc, $rtrgseqn, $width, $sbjct, $sbjctstart, $sbjctend, $sbfasta, $e_val, $$rproc);
            $$rproc++;
        }
    }

    close( OUT );
    print(STDOUT "\nClosest matches to CONSENSUS:\n");
    printf(STDOUT " %s(%d)",$_,$seqmatches{$_}) foreach (sort {$seqmatches{$b} <=> $seqmatches{$a}} keys %seqmatches);
    print(STDOUT "\n\n");
    return 1;
}

sub PrintAlignedHelperPass2
{
    my  $reffile = shift;##file reference
    my  $keepdel = shift;
    my  $rtrgdesc = shift;
    my  $rtrgseqn = shift;
    my  $width = shift;
    my  $sbjct = shift;
    my  $sbjctstart = shift;
    my  $sbjctend = shift;
    my  $sbfasta = shift;
    my  $e_val = shift;
    my  $proc = shift;

    my  $trgfastacopy = $$rtrgseqn;
    my  $sbtfastacopy = $$sbfasta;

    if(!length($sbtfastacopy) || length($trgfastacopy) != length($sbtfastacopy)) {
        printf(STDERR "ERROR: Inconsistent target and template (%s) lengths: %d vs %d\n",
            $sbjct, length($trgfastacopy), length($sbtfastacopy));
        return 0;
    }

    unless($keepdel) {
        for(my $i = length($trgfastacopy)-1; 0 <= $i; $i--) {
            if(substr($trgfastacopy,$i,1) eq '-') {
                substr($trgfastacopy,$i,1,'');
                substr($sbtfastacopy,$i,1,'');
            }
        }
    }

    unless($proc) {
        printf($reffile ">%s\n", $$rtrgdesc );
        WrapFasta( $reffile, \$trgfastacopy, $width );
    }

    $sbjct =~ s/^(\S+)\s*(.*)$/$1 ($sbjctstart-$sbjctend) $2/;
    printf($reffile ">%s\n", $sbjct );
    WrapFasta( $reffile, \$sbtfastacopy, $width );

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

