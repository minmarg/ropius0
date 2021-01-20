#!/usr/bin/perl -w

##
## (C)2019-2020 Mindaugas Margelevicius
## Institute of Biotechnology, Vilnius University
##

use strict;
use FindBin;
use lib "$FindBin::Bin";
use File::Spec;
use File::Basename;
use Getopt::Long;
use Scalar::Util qw(looks_like_number);

my  $MYPROGNAME = basename($0);
my  $MAXRETERMS = 29;##maximum number of Rosetta talaris2014 energy terms per residue
##(fa_atr, fa_elec, hbond_sr_bb, hbond_lr_bb, hbond_bb_sc, hbond_sc, fa_dun, p_aa_pp)
my  @RETERMS = (3,7,9,10,11,12,21,22);##Rosetta talaris2014 energy terms of interest (being considered)
my  $RETERMSlst = join(',', @RETERMS);

my  $MAXERR = 8.0;##max distance error in Angstroms
my  $logMAXERR = log($MAXERR);##log of max distance error in Angstroms

my  $WGT = 0.5;

my  $usage = <<EOIN;

Combine protein structure QA predictions and Rosetta per-residue 
energy scores to produce a final prediction.
Information not related to predictions will be copied from the first 
prediction file.
(C)2019-2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters> <predictionfile1> <predictionfile2> ...

Parameters:

-o <filename>      Output prediction file.

-d <directory>     Directory of Rosetta per-residue energy scores for each 
                   structure present in the prediction files.

-t <list>          Comma-separated Rosetta energy term indices to consider.
           Default=$RETERMSlst

-w [0-1]           Weight with which Rosetta energy scores contribute to 
                   the final prediction.
           Default=$WGT

-h                 This text.

EOIN

##programs:
my  $QAcombine = "$FindBin::Bin/QAcombine.pl";

my  $OUTPUT;
my  $PRESCORESDIR;##directory of Rosetta per-residue energy scores for each structure
my  $Fail = 0;

my  $result = GetOptions(
               'o=s'      => \$OUTPUT,
               'd=s'      => \$PRESCORESDIR,
               't=s'      => \$RETERMSlst,
               'w=f'      => \$WGT,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Output file exists. Leaving to avoid overwriting.\n"; $Fail = 1; } if($OUTPUT && -f $OUTPUT);
do { print STDERR "ERROR: Rosetta energy scores file not found.\n"; $Fail = 1; } unless($Fail || ($PRESCORESDIR && -d $PRESCORESDIR));
do { print STDERR "ERROR: No input files.\n"; $Fail = 1; } unless($Fail || $#ARGV>=0);
do { print STDERR "ERROR: Invalid weight.\n"; $Fail = 1; } unless($Fail || ($WGT>=0 && $WGT<=1));

@RETERMS = split(',',$RETERMSlst);
do {die "ERROR: Invalid energy term indices." unless looks_like_number($_)} foreach @RETERMS;

do { print( STDERR "ERROR: Program not found: $QAcombine\n"); exit(1); } unless($QAcombine && -f $QAcombine);

exit(1) if($Fail);

my (%PREDS, $pred, $F);

my ($ninputs,$skip) = (0,0);
my $cmdline = $QAcombine;
my @rscfiles;
my $output;

ReadFiles($PRESCORESDIR, \@rscfiles);

for(my $ndx = 0; $ndx <= $#ARGV && !$Fail; $ndx++) {
    my $input = $ARGV[$ndx];
    if($input =~ /^\-+[odtw]$/) {
        $skip = 1;
        next;
    }
    do{$skip--; next} if $skip>0;
    unless(-f $input) {
        print(STDERR "ERROR: Input file does not exist: $input\n");
        $Fail = 1;
        last;
    }
    $cmdline .= " ${input}";
    $ninputs++;
}

exit(1) if($Fail);

die "ERROR: No valid inputs found." if($ninputs < 1);

unless( RunCommand($cmdline, 0, \$output)) {
    printf(STDERR "ERROR: Running combined prediction failed.\n");
    exit(1);
}

my @lines = split(/\n/, $output);
my ($ser,$maxnflds)=(0,0);
my ($header,$footer,$body)=('','','');

foreach (@lines) {
    ##if(/^T\d+TS\d+_\d+/) {
    if(/^\S+\s+\d+\.\d+\s+(?:[\d\.]+|X)\s+(?:[\d\.]+|X)/) { ##general name definition
        my @flds = split(/\s+/);
        $maxnflds = $#flds+1 unless $maxnflds;
        if($maxnflds != $#flds+1) {
            printf(STDERR "ERROR: Invalid #fields (=%d) for %s (combined result)\n",$#flds+1,$flds[0]);
            $Fail = 1;
            last;
        }
        if(exists $PREDS{SER}{$flds[0]} || exists $PREDS{GLB}{$flds[0]} ||
           exists $PREDS{LOC}{$flds[0]}) {
            printf(STDERR "ERROR: Prediction already exists for %s (combined result)\n",$flds[0]);
            $Fail = 1;
            last;
        }
        $ser++;
        $PREDS{SER}{$flds[0]} = $ser;
        $PREDS{GLB}{$flds[0]} = $flds[1];
        push @{$PREDS{LOC}{$flds[0]}}, $flds[$_] foreach(2..$#flds);
    }
    else {
        if($ser) { $footer .= "$_\n" }
        else { $header .= "$_\n" }
    }
}

exit(1) if($Fail);

$body .= $header;

foreach $pred (sort {$PREDS{SER}{$a}<=>$PREDS{SER}{$b}} keys %{$PREDS{SER}}) {
    last if $Fail;
    my $input = "$PRESCORESDIR/${pred}.rsc";
    unless(-f $input) {
        print(STDERR "WARNING: Rosetta file of per-residue energies not found: $input\n");
    }
    else {
        unless(open(F, $input)) {
            print(STDERR "ERROR: Failed to open Rosetta file: $input\n");
            $Fail = 1;
            last;
        }
        my $desc = <F>;##skip the first line
        while(<F>) {
            my @flds = split(/\s+/);
            my $sum = 0;
            if($MAXRETERMS != $#flds+1) {
                printf(STDERR "ERROR: Invalid #fields (=%d) for %s (%s)\n",$#flds+1,$pred,$input);
                $Fail = 1;
                last;
            }
            $flds[2] =~ s/[A-Za-z]$//;##NOTE: addedd on 2020-06-12
            my $resnum = $flds[2]-1;
            if($resnum > $#{$PREDS{LOC}{$pred}}) {
                printf(STDERR "ERROR: Invalid residue number (%d) from Rosetta file for %s (%s)\n",$resnum,$pred,$input);
                $Fail = 1;
                last;
            }
            my $pval = $PREDS{LOC}{$pred}[$resnum];
            ##ignore residues for which there's no prediction
            next if $pval =~ /X/;

            $sum += $_ foreach @flds[@RETERMS];
            my $thr = ($MAXERR < $pval)? $pval: $MAXERR;
            my $dst = ($sum < log($thr))? exp($sum): $thr;

            $PREDS{LOC}{$pred}[$resnum] = $pval * (1.0-$WGT) + $dst * $WGT;
        }
        close(F);
    }
    my ($globdst,$c)=(0,0);
    my $locpred = '';
    foreach my $est (@{$PREDS{LOC}{$pred}}) {
        $locpred .= ($est =~ /X/)? ' X': sprintf(" %.1f",$est);
        next if $est =~ /X/;
        $globdst += $est;
        $c++;
    }
    ## divide by 8 to constrain quality estimate within the range [0,1]
    ## (see QA4segm_519.py):
    $globdst = 1.0 - $globdst/($c * 8.0) if $c;
    if($globdst < 0 || 1 < $globdst) {
        printf(STDERR "WARNING: Invalid mean global distance for %s (%g) adjusted.\n",$pred,$globdst);
        $globdst = 0 if $globdst < 0;
        $globdst = 1 if 1 < $globdst;
    }
    $body .= sprintf("%s %.5f  %s\n",$pred,$globdst,$locpred);
}

exit(1) if($Fail);

$body .= $footer;

$F = \*STDOUT unless $OUTPUT;
if($OUTPUT && !open($F, ">", $OUTPUT)) {
    print(STDERR "ERROR: Failed to open file for writing: $OUTPUT\n");
    exit(1);
}
print($F $body);
close($F) if $OUTPUT;

print("Done.\n") if $OUTPUT;
exit(0);


## -------------------------------------------------------------------
## read file list from directory
##
sub ReadFiles {
    my  $dirname = shift;
    my  $refiles = shift;

    opendir( DIR, $dirname ) || die "ERROR: Cannot open directory $dirname.";

    @{$refiles} = grep { -f "$dirname/$_" } readdir( DIR );

    closedir( DIR );
}

## ---------------------------------------------------------------------------
##
sub RunCommand
{
    my  $cmdline = shift;
    my  $retstatus = shift;
    my  $routput = shift;##ref

    if($cmdline) {
        $$routput = `$cmdline 2>&1` if $routput;
        system( "$cmdline" ) unless $routput;
    }

    if( $? == -1 ) {
        printf( STDERR "ERROR: Failed to execute command: $!\n" );
        return 0;
    }
    if( $? & 127 ) {
        printf( STDERR "ERROR: Command terminated with signal %d (%s coredump).\n",
            ($? & 127), ($? & 128)? 'with' : 'without' );
        return 0;
    }
    else {
        if(( $? >> 8 ) != 0 ) {
            unless( $retstatus ) {
                printf( STDERR "ERROR: Command failed and exited with status %d\n", $? >> 8 );
                return 0
            }
            return( $? >> 8 );
        }
    }
    return 1;
}

