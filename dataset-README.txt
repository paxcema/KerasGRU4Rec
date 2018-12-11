SUMMARY
================================================================================

This dataset was constructed by YOOCHOOSE GmbH to support participants in the RecSys Challenge 2015.
See  http://recsys.yoochoose.net for details about the challenge.

The YOOCHOOSE dataset contain a collection of sessions from a retailer, where each session
is encapsulating the click events that the user performed in the session.
For some of the sessions, there are also buy events; means that the session ended
with the user bought something from the web shop. The data was collected during several
months in the year of 2014, reflecting the clicks and purchases performed by the users
of an on-line retailer in Europe.  To protect end users privacy, as well as the retailer,
all numbers have been modified. Do not try to reveal the identity of the retailer.

LICENSE
================================================================================
This dataset is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.
YOOCHOOSE cannot guarantee the completeness and correctness of the data or the validity
of results based on the use of the dataset as it was collected by implicit tracking of a website. 
If you have any further questions or comments, please contact YooChoose <support@YooChoose.com>. 
The data is provided "as it is" and there is no obligation of YOOCHOOSE to correct it,
improve it or to provide additional information about it.

CLICKS DATASET FILE DESCRIPTION
================================================================================
The file yoochoose-clicks.dat comprising the clicks of the users over the items.
Each record/line in the file has the following fields/format: Session ID, Timestamp, Item ID, Category
-Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer number.
-Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
-Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer number.
-Category – the context of the click. The value "S" indicates a special offer, "0" indicates  a missing value, a number between 1 to 12 indicates a real category identifier,
 any other number indicates a brand. E.g. if an item has been clicked in the context of a promotion or special offer then the value will be "S", if the context was a brand i.e BOSCH,
 then the value will be an 8-10 digits number. If the item has been clicked under regular category, i.e. sport, then the value will be a number between 1 to 12. 
 
BUYS DATSET FILE DESCRIPTION
================================================================================
The file yoochoose-buys.dat comprising the buy events of the users over the items.
Each record/line in the file has the following fields: Session ID, Timestamp, Item ID, Price, Quantity

-Session ID - the id of the session. In one session there are one or many buying events. Could be represented as an integer number.
-Timestamp - the time when the buy occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
-Item ID – the unique identifier of item that has been bought. Could be represented as an integer number.
-Price – the price of the item. Could be represented as an integer number.
-Quantity – the quantity in this buying.  Could be represented as an integer number.

TEST DATASET FILE DESCRIPTION
================================================================================
The file yoochoose-test.dat comprising only clicks of users over items.
This file served as a test file in the RecSys challenge 2015. 
The structure is identical to the file yoochoose-clicks.dat but you will not find the
corresponding buying events to these sessions in the yoochoose-buys.dat file.
