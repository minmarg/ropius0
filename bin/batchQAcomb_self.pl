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
my  $TARGETDIR = '/home/mindaugas/projects/ROPIUS0/QA-tests/FM';
##maximum number of predicted structures to evaluate:
my  $NUM_EVAL = 50;

my  $usage = <<EOIN;

Combine protein structure QA predictions and evaluate the final obtained
prediction. The combination is confined to different values of distance and 
probability thresholds.
Prediction summary (evaluation) files can be given by arguments and/or 
their information can be read from STDIN.
(C)2019-2020 Mindaugas Margelevicius, Vilnius University

Usage:
$MYPROGNAME <Parameters> <predictionsum1> <predictionsum2> ...

Parameters:

-o <filename>      Output evaluation file.

-h                 This text.

EOIN

##programs:
my  $QAcombine = "$FindBin::Bin/QAcombine.pl";

my  $OUTPUT;
my  $Fail = 0;

my  $result = GetOptions(
               'o=s'      => \$OUTPUT,
               'help|h'   => sub { print $usage; exit( 0 ); }
);

do { print $usage; $Fail = 1; }  unless $result;
do { print STDERR "ERROR: Output file exists. Leaving to avoid overwriting.\n"; $Fail = 1; } if($OUTPUT && -f $OUTPUT);

do { print( STDERR "ERROR: Program not found: $QAcombine\n"); exit(1); } unless($QAcombine && -f $QAcombine);

my ($ndx,$skip,$maxnflds,$ser)=(0,0,0,0);
my ($header,$footer,$body)=('','','');

my (%SUMMS, $pred, $F);

my $ninputs = 0;

## read STDIN first
exit(1) unless GetModelEvals('', \$maxnflds, \%SUMMS);

## read files given by arguments
for($ndx = 0; $ndx <= $#ARGV && !$Fail; $ndx++) {
    my $input = $ARGV[$ndx];
    if($input =~ /^\-+o$/) {
        $skip = 1;
        next;
    }
    do{$skip--; next} if $skip>0;
    $ninputs++;
    unless( GetModelEvals($input, \$maxnflds, \%SUMMS)) {
        $Fail = 1;
        last;
    }
}

exit(1) if($Fail);

my @models = keys %SUMMS;
my $modelreg = qr/^QA2__modelRUN(\d+)e(\d+)__full_d(\d+)_p(\S+)/;
my ($resline, $finresults);

die "ERROR: No models (at least two) read." if($#models < 1);

printf("\n%d inputs read, excluding STDIN; %d total models\n",$ninputs,$#models+1);

## limit to three levels of model combination
for(my $i1 = 0; $i1 <= $#models && !$Fail; $i1++ ) {
    my $mod1name = $models[$i1];
    my ($mrun1, $mnum1, $dst1, $prb1);
    if( $mod1name =~ /$modelreg/) {
        $mrun1 = $1; $mnum1 = $2;
        $dst1 = $3; $prb1 = $4;
    }
    else {
        printf(STDERR "ERROR: Unrecognized model name for %s\n",$mod1name);
        $Fail = 1;
        last;
    }
    ##{{TEST:
    #unless( GetResultOfCombinedModels(
    #        $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline, \%SUMMS, ($mod1name) )) {
    #    $Fail = 1;
    #    last;
    #}
    #print($resline);
    ##}}
 
    for(my $i2 = $i1+1; $i2 <= $#models && !$Fail; $i2++ ) {
        my $mod2name = $models[$i2];
        my ($mrun2, $mnum2, $dst2, $prb2);
        if( $mod2name =~ /$modelreg/) {
            $mrun2 = $1; $mnum2 = $2;
            $dst2 = $3; $prb2 = $4;
        }
        else {
            printf(STDERR "ERROR: Unrecognized model name for %s\n",$mod2name);
            $Fail = 1;
            last;
        }
        next if($dst2 == $dst1);
        unless( GetResultOfCombinedModels(
                $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                \%SUMMS, ($mod1name,$mod2name) )) {
            $Fail = 1;
            last;
        }
        print($resline);
        $finresults .= $resline;

        for(my $i3 = $i2+1; $i3 <= $#models && !$Fail; $i3++ ) {
            my $mod3name = $models[$i3];
            my ($mrun3, $mnum3, $dst3, $prb3);
            if( $mod3name =~ /$modelreg/) {
                $mrun3 = $1; $mnum3 = $2;
                $dst3 = $3; $prb3 = $4;
            }
            else {
                printf(STDERR "ERROR: Unrecognized model name for %s\n",$mod3name);
                $Fail = 1;
                last;
            }
            next if($dst3 == $dst1 || $dst3 == $dst2);

            unless( GetResultOfCombinedModels(
                    $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                    \%SUMMS, ($mod1name,$mod2name,$mod3name) )) {
                $Fail = 1;
                last;
            }
            print($resline);
            $finresults .= $resline;

            for(my $i4 = $i3+1; $i4 <= $#models && !$Fail; $i4++ ) {
                my $mod4name = $models[$i4];
                my ($mrun4, $mnum4, $dst4, $prb4);
                if( $mod4name =~ /$modelreg/) {
                    $mrun4 = $1; $mnum4 = $2;
                    $dst4 = $3; $prb4 = $4;
                }
                else {
                    printf(STDERR "ERROR: Unrecognized model name for %s\n",$mod4name);
                    $Fail = 1;
                    last;
                }
                next if($dst4 == $dst1 || $dst4 == $dst2 || $dst4 == $dst3);

                unless( GetResultOfCombinedModels(
                        $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                        \%SUMMS, ($mod1name,$mod2name,$mod3name,$mod4name) )) {
                    $Fail = 1;
                    last;
                }
                print($resline);
                $finresults .= $resline;

                for(my $i5 = $i4+1; $i5 <= $#models && !$Fail; $i5++ ) {
                    my $mod5name = $models[$i5];
                    my ($mrun5, $mnum5, $dst5, $prb5);
                    if( $mod5name =~ /$modelreg/) {
                        $mrun5 = $1; $mnum5 = $2;
                        $dst5 = $3; $prb5 = $4;
                    }
                    else {
                        printf(STDERR "ERROR: Unrecognized model name for %s\n",$mod5name);
                        $Fail = 1;
                        last;
                    }
                    next if($dst5 == $dst1 || $dst5 == $dst2 || $dst5 == $dst3 || $dst5 == $dst4);

                    unless( GetResultOfCombinedModels(
                            $QAcombine, $TARGETDIR, $NUM_EVAL, \$resline,
                            \%SUMMS, ($mod1name,$mod2name,$mod3name,$mod4name,$mod5name) )) {
                        $Fail = 1;
                        last;
                    }
                    print($resline);
                    $finresults .= $resline;
                }
            }
        }
    }
}


exit(1) if($Fail);

$F = \*STDOUT unless $OUTPUT;
if($OUTPUT && !open($F, ">", $OUTPUT)) {
    print(STDERR "ERROR: Failed to open file for writing: $OUTPUT\n");
    exit(1);
}
print($F $finresults) if $OUTPUT;
close($F) if $OUTPUT;

print("Done.\n") if $OUTPUT;
exit(0);

## ---------------------------------------------------------------------------

sub GetModelEvals
{
    my $filename = shift;
    my $rmaxnflds = shift;
    my $rsumms = shift;
    my $ret = 1;
    my ($fh, $inputname);
    ##
    $inputname = $filename? $filename: 'STDIN';
    $fh = \*STDIN unless $filename;
    if($filename && !open($fh, $filename)) {
        print(STDERR "ERROR: GetModelEvals: Failed to open file: $filename\n");
        return 0;
    }
    while(<$fh>) {
        my @flds = split(/\s+/);
        $$rmaxnflds = $#flds+1 unless $$rmaxnflds;
        if($$rmaxnflds != $#flds+1) {
            printf(STDERR "ERROR: GetModelEvals: Invalid #fields (=%d) for %s (%s)\n",
                $#flds+1,$flds[0],$inputname);
            $ret = 0;
            last;
        }
        if(exists $$rsumms{$flds[0]}) {
            printf(STDERR "ERROR: GetModelEvals: Model already exists for %s (%s)\n",
                $flds[0],$inputname);
            $ret = 0;
            last;
        }
        do{push @{$$rsumms{$flds[0]}{TRG}}, $flds[$_] if $flds[$_]=~/^T/} for(4..$#flds);
    }
    close($fh) if $filename;
    return $ret;
}

## ---------------------------------------------------------------------------

sub GetResultOfCombinedModels
{
    my $program = shift;##program to combine predictions
    my $targetdir = shift;##directory of targets
    my $num_eval = shift;##number of predictions to evaluate
    my $rresults = shift;##ref to results string
    my $rsumms = shift;##ref to summary evaluation results hash
    my @modelids = @_;##model (evaluated) ids
    my $ret = 1;
    my ($trg, $mod, $combname, $cmdline, $output);
    my ($ln, $evalresults, @means);
    my ($c,$mean,$sd)=(0,0,0);
    my $ntargets = 0;
    if( $#modelids < 0 ) {
        printf(STDERR "ERROR: GetResultOfCombinedModels: Invalid #models (=%d)\n",
            $#modelids+1);
        return 0;
    }
    foreach $trg (@{$$rsumms{$modelids[0]}{TRG}}) {
        $cmdline = $program;
        foreach $mod (@modelids) {
            if( $#{$$rsumms{$mod}{TRG}} != $#{$$rsumms{$modelids[0]}{TRG}}) {
                printf(STDERR "ERROR: GetResultOfCombinedModels: Inconsistent number of targets (=%d) for %s\n",
                    $#{$$rsumms{$mod}{TRG}}, $mod);
                return 0;
            }
            my $modfilename = "${targetdir}/${trg}/QA2--${trg}/${trg}_${mod}";
            unless(-f $modfilename) {
                printf(STDERR "ERROR: GetResultOfCombinedModels: Model results file not found: %s\n",
                    $modfilename);
                return 0;
            }
            $cmdline .= " ${modfilename}";
        }

        unless( RunCommand($cmdline, 0, \$output)) {
            printf(STDERR "ERROR: GetResultOfCombinedModels: Running combined prediction failed for %s\n",$mod);
            return 0;
        }

        ##{{ground truth
        my $gt_file = "${targetdir}/${trg}/${trg}.txt"; ##ground truth file
        my %gt_trgs;
        unless( open(F, $gt_file)) {
            printf(STDERR "ERROR: Failed to open $gt_file\n");
            return 0;
        }
        while(<F>) { $gt_trgs{$2} = $1 if /^\s*(\d+)\s+([^\-\s]+)/ };
        close(F);
        ##}}

        my @lines = grep(/^${trg}/, split(/\n/, $output));
        ($c,$mean,$sd)=(0,0,0);
        my @mae;
        foreach $ln (sort {(split(/\s+/,$b))[1] <=> (split(/\s+/,$a))[1]} @lines) {
            my @a = split(/\s+/,$ln);
            unless( exists $gt_trgs{$a[0]}) {
                printf(STDERR "ERROR: Target model %s not found in ground truth.\n",$a[0]);
            }
            $c++;
            push @mae, abs($gt_trgs{$a[0]}-$c);
            last if $num_eval <= $c;
        }
        unless($c) {
            printf(STDERR "ERROR: No evaluation points for target %s (%s)\n",$trg,join(',',@modelids));
            return 0;
        }
        $mean += $_ foreach @mae; $mean /= $c;
        $sd += ($_-$mean)*($_-$mean) foreach @mae; $sd = sqrt($sd/$c);
        $evalresults .= sprintf("   %-5s %7.3f %7.3f",${trg},$mean,$sd);
        push @means, $mean;
        $ntargets++;
    }
    if($#means < 0) {
        printf(STDERR "ERROR: No MAE values over targets obtained (%s)\n",join(',',@modelids));
        return 0;
    }

    ($c,$mean,$sd)=($ntargets,0,0);
    $mean += $_ foreach @means; $mean /= $c;
    $sd += ($_-$mean)*($_-$mean) foreach @means; $sd = sqrt($sd/$c);

    $combname = join('__',@modelids);
    $$rresults = sprintf("%-200s %7.3f %7.3f |%s\n",$combname,$mean,$sd,$evalresults);
    return $ret;
}

## ---------------------------------------------------------------------------

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

