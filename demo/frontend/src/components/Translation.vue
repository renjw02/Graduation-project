<template>
  <div>
    <h1>NL2PQL</h1>
    <el-row>
      <el-col :span="18">
        <h3>自然语言问题</h3>
        <el-input v-model="question" placeholder="请输入自然语言问题" type="textarea" :rows="4"></el-input>
      </el-col>
    </el-row>
    <el-row style="margin-top: 20px;">
      <el-col :span="6">
        <h3>选择数据集</h3>
        <el-select v-model="selectedDatabase" placeholder="选择数据集" size="large">
          <el-option v-for="db in databases" :key="db.value" :label="db.label" :value="db.value"></el-option>
        </el-select>
      </el-col>
      <!-- <el-col :span="6" style="margin-left: 20px;">
        <el-upload :action="importDatabaseUrl" :on-success="handleImportSuccess" :on-error="handleImportError">
          <el-button slot="trigger" type="primary">导入数据集</el-button>
        </el-upload>
      </el-col> -->
    </el-row>
    <el-row style="margin-top: 20px;">
      <el-col :span="3">
        <el-button type="primary" @click="generateQuery" :disabled="disabledQueryButton">生成查询</el-button>
      </el-col>
      <el-col :span="3">
        <el-button type="primary" @click="resetAll">清空</el-button>
      </el-col>
    </el-row>
    <el-row style="margin-top: 20px;">
      <el-col :span="12">
        <div v-if="query">
          <h3>查询语句：</h3>
          <pre>{{ query }}</pre>
        </div>
        <div v-if="success">
          <div v-if="costTime > 0">
            <h3>生成耗时：</h3>
            <p>{{ costTime }} s</p>
          </div>
          <div v-if="cols.length > 0">
            <h3>查询结果：</h3>
            <el-table :data="results" border style="width: 100%">
              <el-table-column v-for="(column, index) in cols" :key="index" :prop="column"
                :label="column"></el-table-column>
            </el-table>
          </div>
        </div>
        <div v-else>
          <h3>查询失败</h3>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { ref } from "vue";
import { ElButton, ElCol, ElInput, ElOption, ElRow, ElSelect, ElTable, ElTableColumn, ElUpload } from "element-plus";
import axios from "axios";

export default {
  name: "Translation",
  components: {
    ElButton,
    ElCol,
    ElInput,
    ElOption,
    ElRow,
    ElSelect,
    ElTable,
    ElTableColumn,
    ElUpload,
  },
  setup() {
    const question = ref("");
    const selectedDatabase = ref("");
    const databases = ref([
      { label: "WikiSQL", value: "wikisql" },
      { label: "Advising", value: "advising" },
      { label: "Restaurants", value: "restaurants" },
      { label: "IMDB", value: "imdb" },
    ]);
    const importDatabaseUrl = ref("/import-database");
    const query = ref("");
    const results = ref([]);
    const rows = ref([]);
    const cols = ref([]);
    const costTime = ref(0);
    const success = ref(true);
    const disabledQueryButton = ref(false);

    async function generateQuery() {
      disabledQueryButton.value = true;
      reset();

      if (selectedDatabase.value !== "wikisql" && selectedDatabase.value !== "advising" && selectedDatabase.value !== "restaurants" && selectedDatabase.value !== "imdb") {
        console.error("Invalid database selected:", selectedDatabase.value);
        return;
      }

      // 在这里实现生成查询语句的逻辑
      await axios.post("http://192.168.215.36:5000/predict", {
        question: question.value,
        dataset: selectedDatabase.value,
      }).then((response) => {
        query.value = response.data.pql;
        rows.value = response.data.results;
        cols.value = response.data.cols;
        costTime.value = response.data.cost_time.toFixed(3);
        success.value = response.data.success;
      }).catch((error) => {
        console.error("Failed to generate query:", error);
      });

      console.log("rows:", rows.value);
      console.log("cols:", cols.value);
      disabledQueryButton.value = false;
      if (!success.value) {
        return;
      }

      // 在这里实现执行查询并更新结果的逻辑
      for (let i = 0; i < rows.value.length; i++) {
        const result = {};
        for (let j = 0; j < cols.value.length; j++) {
          result[cols.value[j]] = rows.value[i][j];
        }
        results.value.push(result);
      }

      // console.log("Query results:", results.value);
    }

    function handleImportSuccess(response) {
      // 在这里实现导入数据库成功的逻辑
      console.log("Database imported successfully:", response);
    }

    function handleImportError(error) {
      // 在这里实现导入数据库失败的逻辑
      console.error("Failed to import database:", error);
    }

    function reset() {
      query.value = "";
      results.value = [];
      costTime.value = 0;
      rows.value = [];
      cols.value = [];
      success.value = true;
    }

    function resetAll() {
      question.value = "";
      selectedDatabase.value = "";
      reset();
    }

    return {
      question,
      selectedDatabase,
      databases,
      importDatabaseUrl,
      query,
      results,
      costTime,
      success,
      cols,
      generateQuery,
      handleImportSuccess,
      handleImportError,
      resetAll,
      disabledQueryButton,
    };
  },
};
</script>

<style scoped>
/* 在这里添加样式 */
pre {
  font-size: large;
  white-space: pre-wrap;       /* Since CSS 2.1 */
  word-wrap: break-word;       /* Internet Explorer 5.5+ */
  overflow: auto;              /* 不剪裁内容 */
  -webkit-overflow-scrolling: touch; /* 允许在iOS设备上滚动 */
}

.bold-header {
  font-weight: bold;
  color: black;
}
</style>