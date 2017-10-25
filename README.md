# Temporal causality in social network cascades

The main drawback of regression based continuous modeling
of time series to deduce cause and effect relationships is that 
(1) Firstly, it is difficult to extract specific time time intervals within X (the regressor)
as intervals which carry the most significant information in
determining what effect it has on Y (the response variable) in the next few time points. Traditionally,
Granger causal models are used to study dependency between
different features (2) Secondly, there are no specific
methods to measure the "level" of the significance as far as the different
regression models are concerned other than the regular statistical testing.
Non-parametric causality has been studied in the form of propositional
connectives and boolean logical operators to define causal
relationships between different features and the response variable and hnece address the 
above shortcomings.

The code demonstrates an example of using the Kleinger causality framework used in "The Temporal Logic of Causal Structures" by Smantha Kleinberg et. al 2012. The code and ideas are entirely mine although adopted from that paper and in no way is an authority on the concept used in that paper.
