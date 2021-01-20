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

my  $MYPROGNAME = basename($0);

my  $usage = <<EOIN;

Combine protein structure QA predictions to produce a final prediction.
Information not related to predictions will be copied from the first 
prediction file.
(C)2019-2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters> <predictionfile1> <predictionfile2> ...

Parameters:

-o <filename>      Output prediction file.

-h                 This text.

EOIN

my  $OUTPUT;
my  $Fail = 0;

my  $result = GetOptions(
               'o=s'      => \$OUTPUT,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Output file exists. Leaving to avoid overwriting.\n"; $Fail = 1; } if($OUTPUT && -f $OUTPUT);
do { print STDERR "ERROR: No input files.\n"; $Fail = 1; } unless($Fail || $#ARGV>=0);

my ($ndx,$skip,$maxnflds,$ser)=(0,0,0,0);
my ($header,$footer,$body)=('','','');

my (%PREDS, $pred, $F);

my $ninputs = 0;

for($ndx = 0; $ndx <= $#ARGV && !$Fail; $ndx++) {
    my $input = $ARGV[$ndx];
    if($input =~ /^\-+o$/) {
        $skip = 1;
        next;
    }
    do{$skip--; next} if $skip>0;
    unless(-f $input) {
        print(STDERR "ERROR: Input file does not exist: $input\n");
        $Fail = 1;
        last;
    }
    unless(open(F, $input)) {
        print(STDERR "ERROR: Failed to open input file: $input\n");
        $Fail = 1;
        last;
    }
    $ninputs++;
    $ser = 0;
    while(<F>) {
        if(/^T\d+TS\d+_\d+/) {
            my @flds = split(/\s+/);
            $maxnflds = $#flds+1 unless $maxnflds;
            if($maxnflds != $#flds+1) {
                printf(STDERR "ERROR: Invalid #fields (=%d) for %s (%s)\n",$#flds+1,$flds[0],$input);
                $Fail = 1;
                last;
            }
            if(exists $PREDS{SER}{$flds[0]}[$ndx] || exists $PREDS{GLB}{$flds[0]}[$ndx] ||
               exists $PREDS{LOC}{$flds[0]}[$ndx]) {
                printf(STDERR "ERROR: Prediction already exists for %s (%s)\n",$flds[0],$input);
                $Fail = 1;
                last;
            }
            $ser++;
            $PREDS{SER}{$flds[0]}[$ndx] = $ser;
            $PREDS{GLB}{$flds[0]}[$ndx] = $flds[1];
            push @{$PREDS{LOC}{$flds[0]}[$ndx]}, $flds[$_] foreach(2..$#flds);
        }
        elsif($ndx < 1) {
            if($ser) { $footer .= $_ }
            else { $header .= $_ }
        }
    }
    close(F);
}

exit(1) if($Fail);

die "ERROR: No valid inputs found." if($ninputs < 1);

$body .= $header;

foreach $pred (sort {$PREDS{SER}{$a}[0]<=>$PREDS{SER}{$b}[0]} keys %{$PREDS{SER}}) {
    last if $Fail;
    my ($gsum,$c)=(0,0);
    for($ndx = 0; $ndx < $ninputs && !$Fail; $ndx++) {
        unless(exists $PREDS{LOC}{$pred}[$ndx]) {
            printf(STDERR "ERROR: Prediction %s not found for input %d\n",$pred,$ndx);
            $Fail = 1;
            last;
        }
        $gsum += $PREDS{GLB}{$pred}[$ndx];
        $c++;
    }
    if($c != $ninputs) {
        printf(STDERR "ERROR: Prediction %s was unavailable for some of the inputs\n",$pred);
        $Fail = 1;
        last;
    }
    $body .= sprintf("%s %.5f  ",$pred,$gsum/$c);
    for(my $i = 0; $i < $maxnflds-2 && !$Fail; $i++) {
        my ($sum,$c)=(0,0);
        for($ndx = 0; $ndx < $ninputs && !$Fail; $ndx++) {
            next if $PREDS{LOC}{$pred}[$ndx][$i] =~ /X/;
            $sum += $PREDS{LOC}{$pred}[$ndx][$i];
            $c++;
        }
        if($c < 1) { $body .= ' X' }
        else { $body .= sprintf(" %.1f",$sum/$c) }
    }
    $body .= "\n";
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

