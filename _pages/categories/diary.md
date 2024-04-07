---
title: "Diary"
layout: archive
permalink: /categories/diary
author_profile: true
types: posts
---

{% assign posts = site.categories["diary"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}