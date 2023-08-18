---
layout: post
title: "Modeling a Graph Database on DynamoDB"
date: 2023-08-15
categories: Software
---


## Introduction

I'm recently been fascinated by graphs. They're a simple but powerful data structure to model data and relationships. It's well known that social media companies use graphs to model relationships like friends, followers, etc. In this article I explore graph data modeling by building a graph database on top of DynamoDB. 

# Preliminary

Graphs are made up of nodes and connections. In a graph database, objects can be represented as nodes, and the relationships between objects can be represented as connections. For example, in the diagram below, we define three user nodes (*userA*, *userB*, and *userC*) and a *FRIEND* connection between *userA* and *userB*.

{:refdef: style="text-align: center;"}
![yay](/assets/MGDOD/userFriend.png)
{: refdef}

This structure allows us to answer questions like "Who are *UserA*'s friends?". In a hypothetical graph database, you can answer this question by finding *UserA* and simply retrieving all the *FRIEND* connections to find their friends. So, how do we model this on top of DynamoDB? It turns out to be quite straightforward.

# DynamoDB

DynamoDB is Amazon's proprietary NoSQL database. It can scale virtually limitlessly and is highly available for near real-time applications. Additionally, its on-demand pricing, combined with the Amazon free tier, makes it relatively inexpensive to run small applications. DynamoDB is very flexible in terms of how users can model their data. However, this is a double-edged sword, as modeling data well is essential for optimal utilization. Incorrect modeling can lead to significant costs. The relatively open nature of DynamoDB allows us to construct data structures like a graph on top of it. With that in mind, let's start modeling a graph on DynamoDB.

### Graph Modeling

# Nodes

If you are familiar with DynamoDB, you should know that every item in the database can be addressed using a unique key that consists of a partition key and a sort key. This makes it easy to define our graph nodes. Suppose we have a user with the following attributes:



```
{
    username: ringBearer,
    firstName: Frodo,
    lastName: Baggins,
}
```

This user can be represented as a graph node with the attributes *username*, *firstName* and *lastName* in dynamodb like below.

| PartitionKey | Sortkey | username | firstName | lastName |
| ---- | --- | --- | --- |
| | | ringBearer | Frodo | Baggins |


We will get back to defining the partition Key and sort key later for nodes later.

# Edges

For edges, we can follow a similar strategy to the one discussed above. For example, we might have an edge or a relationship with the following attribute that defines when the relationship was created:

```
{
    createdDate: UNKNOWN
}
```

This edge can be represented in dynamodb like below.

| PartitionKey | Sortkey | createdDate |
| ---- | --- | --- |
| | | UNKNOWN |


### Keys

# Partition Key

Let's now discuss the keys for this table. The real trick to transforming these items into a graph is in defining the keys correctly. The most important requirement in graph databases is the ability to query a node and query all the edges that exist on that node. This requirement can be met by defining the same partition key for a node and its edges. As a bonus, this places the nodes and their edges in the same partition in DynamoDB. Now, we can query a node and all its relations by simply querying the partition key. For our example, we can define a partition key as "USER" like shown below:


| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER | | ringBearer | Frodo | Baggins |
| USER | | | | | UNKNOWN |

# Sort Keys

Next, we need to define the sort keys. The DynamoDB sort keys determine the order of the data in the partition. Therefore, this key can be used to construct hierarchical keys to naturally sort the data (COUNTRY/STATE/CITY/etc.). This also provides an excellent way to group edges by edge type! If we have an edge type of "FRIEND", we can define it as the sort key for the edge.

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER | | ringBearer | Frodo | Baggins |
| USER | FRIEND | | | | UNKNOWN |


This is great but still incomplete. Although, we've defined an edge, it currently leads to nowhere! A diagram of the table as a graph ican be seen below.

{:refdef: style="text-align: center;"}
![yay](/assets/MGDOD/userFriendDB.svg)
{: refdef}

Let's fill out the graph a bit more by adding more users so Frodo can have friends!

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER | | ringBearer | Frodo | Baggins |
| USER | FRIEND | | | | UNKNOWN |
| USER | | theBrave | Samwise | Gamgee |
| USER | | theWhite | Gandalf | |


But now, there is a potential problem with the partition key. We want to make sure that a single partition key always refers to a single node. This means we need to make the partition key unique to the node. There are many ways to ensure that the key is unique. For example, we can include the username as part of the partition key: **USER#{username}**.

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER#Frodo | | ringBearer | Frodo | Baggins |
| USER#Frodo | FRIEND | | | | UNKNOWN |
| USER#Samwise | | theBrave | Samwise | Gamgee |
| USER#Gandalf | | theWhite | Gandalf | |

Now that we have more users, we can start adding friend relationships. We want to make Frodo friends with both Samwise and Gandalf. You might be tempted to simply create a new attribute on the edge and add all of the partition keys of the friends, like below:

```
{
    PartitionKey: USER#Frodo,
    SortKey: FRIEND,
    friends: [
        {
            PartitionKey: USER#Samwise,
            createDate: UNKNOWN
        },
        {
            PartitionKey: USER#Gandalf,
            createdDate: 3004
        }
    ]
}
```

However, there is a problem with this strategy. The length of the *friends* attribute cannot be very large due to limits in DynamoDB. Additionally, you cannot know if two users are friends without searching the entire list for the user's partition key. A better approach would be to simply add multiple *FRIEND* relations with a unique sort key. We can define the relation's sort key as **FRIEND-{partitionKey}**, where the partition key is the key the relation points to. Now, we can define multiple friends like below!



| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER#Frodo | | ringBearer | Frodo | Baggins |
| USER#Frodo | FRIEND-USER#Gandalf | | | | 3004 |
| USER#Frodo | FRIEND-USER#Samwise | | | | UNKNOWN |
| USER#Samwise | | theBrave | Samwise | Gamgee |
| USER#Samwise | FRIEND-USER#Gandalf | | | | UNKNOWN |
| USER#Gandalf | | theWhite | Gandalf | |

Now, the table is looks more like a graph.

{:refdef: style="text-align: center;"}
![yay](/assets/MGDOD/frodoFriends.svg)
{: refdef}

Finally, we need to define the sort key for the nodes. How can we make sure that this key is unique for all possible sort keys in the partition? Simply use the same partition key! The table will finally look like the following:

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER#Frodo | FRIEND-USER#Gandalf | | | | 3004 |
| USER#Frodo | FRIEND-USER#Samwise | | | | UNKNOWN |
| USER#Frodo | USER#Frodo | ringBearer | Frodo | Baggins |
| USER#Samwise | FRIEND-USER#Gandalf | | | | UNKNOWN |
| USER#Samwise | USER#Samwise | theBrave | Samwise | Gamgee |
| USER#Gandalf | USER#Gandalf | theWhite | Gandalf | |

## Queries

At last, the table has enough items in it to perform some read operations! If we wanted to read the attributes for Frodo, we can simply get the item and the attributes using the Boto3 library like below:

```
import boto3
db_client = boto3.client("dynamodb")
db_client.get_item(
    Key={
        "PartitionKey": {"S": "USER#Frodo"},
        "SortKey": {"S": "USER#Frodo}
    },
    ProjectionExpression: "username,firstName,lastName,createdDate"
)
```

If we want to find all of Frodo's friends, we can use the query operation!

```
import boto3
db_client = boto3.client("dynamodb")
resp = db_client.query(
    KeyConditionExpression="PartitionKey = :field1 AND BEGINS_WITH(SortKey, :field2)",
    ExpressionAttributeValues={
        ":field1": {"S": "USER#Frodo"},
        ":field2": {"S": "FRIEND-"}
    }
)
```

Because this query is done under the condition that the sort key begins with *FRIEND*, it will return all items under the partition key **USER#Frodo** whose sort key begins with *FRIEND*. Thus, we can return all of Frodo's friends! Because the sort key contains the partition key of the node that the edge points to, we can use this value to query the friend nodes themselves!

```
[
    {
        "PartitionKey": {S: "USER#Frodo"},
        "SortKey": {S: "FRIEND-USER#Samwise"},
        "createdDate": {S: "UNKNOWN"}
    },
    {
        "PartitionKey": {S: "USER#Frodo"},
        "SortKey": {S: "FRIEND-USER#Gandalf"},
        "createdDate": {S: "3004"}
    }
]
```

### Adding More Nodes

This modeling technique is incredibly versatile because it can be extended very easily. For example, we can start adding other objects to our graph as nodes without impacting our existing users. If we wanted to add places to our graph, we can add these nodes by simply defining a unique key. An example of the table with these new nodes is given below. For clarity, I have removed the partition key for items in the same partition.

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER#Frodo | FRIEND-USER#Gandalf | | | | 3004 |
| | FRIEND-USER#Samwise | | | | UNKNOWN |
| | USER#Frodo | ringBearer | Frodo | Baggins |
| USER#Samwise | FRIEND-USER#Gandalf | | | | UNKNOWN |
|| USER#Samwise | theBrave | Samwise | Gamgee |
| USER#Gandalf | USER#Gandalf | theWhite | Gandalf | |
| **PLACE#TheShire** | **PLACE#TheShire** | | | | |
| **PLACE#Gondor** | **PLACE#Gondor** | | | | |

### Adding More Relationships

We can even start adding new relationships. For example, suppose we wanted to add a *VISITED* relationship. Then, we can simply write the relationships into the table as shown below.

| PartitionKey | Sortkey | username | firstName | lastName | createdDate |
| ---- | --- | --- | --- | --- |
| USER#Frodo | FRIEND-USER#Gandalf | | | | 3004 |
| | FRIEND-USER#Samwise | | | | UNKNOWN |
| | USER#Frodo | ringBearer | Frodo | Baggins |
| | **VISITED-PLACE#Gondor** | | | |
| | **VISITED-PLACE#TheShire** | | | |
| USER#Samwise | FRIEND-USER#Gandalf | | | | UNKNOWN |
| | **VISITED-PLACE#Gondor** ||||
| | **VISITED-PLACE#TheShire** ||||
| | USER#Samwise | theBrave | Samwise | Gamgee |
| USER#Gandalf | USER#Gandalf | theWhite | Gandalf | |
| | **VISITED-PLACE#Gondor** | | | |
| | **VISITED-PLACE#TheShire** | | | |
| PLACE#TheShire | PLACE#TheShire | | | | |
| PLACE#Gondor | PLACE#Gondor | | | | |


Now the graph looks like below.

{:refdef: style="text-align: center;"}
![yay](/assets/MGDOD/allGraph.svg)
{: refdef}

### Inverse Relationships

So far, the table only contains directed relationships. For example, **USER#Frodo FRIENDS-USER#Gandalf** indicates that Frodo is friends with Gandalf, but it does not indicate that Gandalf is friends with Frodo. If we want to store the inverse relationship, we would need to add those relationships to the table as well. For instance, we can add a *VISITED_BY* relation to indicate all the users who have visited a place. The edge, **PLACE#Gondor-VISITED_BY-USER#Frodo** indicates that Gondor was visited by Frodo.


### Secondary Index

Let's take a quick look at what happens when we want to make queries that answer questions about the edges themselves. For example, let's assume that for this data, the *FRIEND* relationship is a one-way relationship (Frodo can be friends with Gandalf, but Gandalf does not consider Frodo a friend). In this scenario, how can we find out all the users who think Gandalf is their friend? This is a difficult question to answer because the relationship we need to query exists on different partitions. However, we can use DynamoDB's secondary index feature to support this query!

Suppose we construct a second index with the sort key as the partition key. Now, the table will look like the following. Note that in DynamoDB, we define which attributes (besides the keys) to project into the secondary index.


| Sortkey | PartitionKey |
| ---- | --- |
| FRIEND-USER#Gandalf | USER#Frodo |
| | USER#Samwise |
| FRIEND-USER#Samwise | USER#Frodo |
| VISITED-PLACE#Gondor | USER#Frodo |
| | USER#Gandalf |
| | USER#Samwise |
| VISITED-PLACE#TheShire |  USER#Frodo |
| | USER#Gandalf |
| | USER#Samwise |
| PLACE#TheShire | PLACE#TheShire |
| PLACE#Gondor | PLACE#Gondor |
| USER#Frodo | USER#Frodo |
| USER#Samwise | USER#Samwise |
| USER#Gandalf | USER#Gandalf |

We can now answer the question by simply querying the edge and retrieving the items in the partition. If we query **FRIEND-USER#Gandalf**, we will get the following items.

```
Items = [
    {
        PartitionKey: {"S": "USER#Frodo"}
    },
    {
        PartitionKey: {"S": "USER#Samwise"}
    }
]
```

We can take this even further by introducing a new sort key. For example, let's say we want to query the edges in a specific order, such as by the date the edge was created (*createdDate*). In this case, we can use the createdDate attribute as a sort key in our secondary index. If we do this, the secondary index table will look like the following:

| Sortkey | createdDate | PartitionKey |
| ---- | --- | --- |
| FRIEND-USER#Gandalf | 3004 | USER#Frodo |
| | UNKNOWN | USER#Samwise |
| FRIEND-USER#Samwise | UNKNOWN | USER#Frodo |

Now when we query **FRIEND-USER#Gandalf**, we will return the edges in the order defined by *createdDate*. Thus we can find out when the user became friends with Gandalf!


## Conclusion

In this article, we discussed how to model a graph database on top of DynamoDB. We saw how the partition key and sort key can be defined to structure a graph in DynamoDB. We also briefly looked at how you can query the nodes' attributes and the edges on those nodes in DynamoDB. Furthermore, we touched on how to include secondary indexes to make more complex queries on the relations. All of this demonstrates how this modeling technique is highly adaptable, as it can be expanded to include new types of nodes, and relations.